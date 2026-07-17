from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.coghp_network import MixerBlock
from utils.networks import MLP, LengthNormalize


class HCoGHPPolicyNetwork(nn.Module):
    """HRM-style hierarchical token mixer for CoGHP.

    The external contract matches `HierarchicalPolicyNetwork`: given observation
    and final goal tokens, it autoregressively predicts latent subgoals followed
    by the primitive action. Internally, each autoregressive generation step uses
    shared low/high MLP-Mixer blocks:

        for each H cycle:
            repeat L cycle updates with shared L block
            update H state with shared H block

    The high state is the only state read by the actor heads.
    """

    num_tokens: int
    state_dim: int
    num_action_dims: int
    joint_embed_dim: int = 128
    num_mixer_blocks: int = 1
    mixer_token_hidden: int = 64
    mixer_channel_hidden: int = 64
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    final_fc_init_scale: float = 1e-2

    high_actor_head: nn.Module = None
    low_actor_head: nn.Module = None
    enc_hidden: Sequence[int] = (128, 128)

    num_subgoals: int = 1
    hrm_l_cycles: int = 2
    hrm_mix_scale: bool = True

    def setup(self):
        self.prev_tokens = self.param(
            'prev_tokens',
            nn.initializers.normal(stddev=0.1),
            (1, self.num_subgoals + 1, self.state_dim),
        )

        num_tokens = self.num_subgoals + 3
        self.low_mixer_block = MixerBlock(
            num_tokens=num_tokens,
            embed_dim=self.state_dim,
            hidden_dim_tokens=self.mixer_token_hidden,
            hidden_dim_channels=self.mixer_channel_hidden,
        )
        self.high_mixer_block = MixerBlock(
            num_tokens=num_tokens,
            embed_dim=self.state_dim,
            hidden_dim_tokens=self.mixer_token_hidden,
            hidden_dim_channels=self.mixer_channel_hidden,
        )

        feature_embed = [
            MLP(hidden_dims=(*self.enc_hidden, self.state_dim), activate_final=False, layer_norm=True),
            LengthNormalize(),
        ]
        self.feature_embed = nn.Sequential(feature_embed)

    def mix_states(self, x_base, z_low, z_high):
        mixed = x_base + z_low + z_high
        if self.hrm_mix_scale:
            mixed = mixed * (1.0 / jnp.sqrt(3.0))
        return mixed

    @staticmethod
    def tensor_norm(x, eps=1e-8):
        return jnp.linalg.norm(x, axis=-1)

    @staticmethod
    def mean_relative_update(new, old, eps=1e-8):
        return jnp.mean(jnp.linalg.norm(new - old, axis=-1) / (jnp.linalg.norm(old, axis=-1) + eps))

    @staticmethod
    def average_diagnostics(diagnostics_list):
        if not diagnostics_list:
            return {}
        keys = diagnostics_list[0].keys()
        return {key: jnp.mean(jnp.stack([diagnostics[key] for diagnostics in diagnostics_list])) for key in keys}

    def hcoghp_mix(self, x_base, return_diagnostics: bool = False):
        z_low = x_base
        z_high = x_base
        diagnostics = {} if return_diagnostics else None

        for h_idx in range(self.num_mixer_blocks):
            for l_idx in range(self.hrm_l_cycles):
                old_z_low = z_low
                z_low = self.low_mixer_block(self.mix_states(x_base, z_low, z_high))
                if return_diagnostics:
                    diagnostics[f'l_update_{l_idx}'] = diagnostics.get(f'l_update_{l_idx}', 0.0) + self.mean_relative_update(
                        z_low, old_z_low
                    )
            old_z_high = z_high
            z_high = self.high_mixer_block(self.mix_states(x_base, z_low, z_high))
            if return_diagnostics:
                diagnostics[f'h_update_{h_idx}'] = self.mean_relative_update(z_high, old_z_high)

        if return_diagnostics:
            for l_idx in range(self.hrm_l_cycles):
                diagnostics[f'l_update_{l_idx}'] = diagnostics[f'l_update_{l_idx}'] / self.num_mixer_blocks
            z_low_norm = self.tensor_norm(z_low)
            z_high_norm = self.tensor_norm(z_high)
            cosine = jnp.sum(z_low * z_high, axis=-1) / (z_low_norm * z_high_norm + 1e-8)
            diagnostics.update(
                {
                    'hrm_cosine_h_l': jnp.mean(cosine),
                    'hrm_relative_distance_h_l': jnp.mean(
                        jnp.linalg.norm(z_high - z_low, axis=-1) / (z_high_norm + 1e-8)
                    ),
                    'z_low_norm': jnp.mean(z_low_norm),
                    'z_high_norm': jnp.mean(z_high_norm),
                    'x_base_norm': jnp.mean(self.tensor_norm(x_base)),
                }
            )
            return z_high, diagnostics

        return z_high

    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: int = None,
        subgoal_reps: Optional[jnp.ndarray] = None,
        action_seq: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        scheduled_sampling_prob: float = 0.0,
        scheduled_sampling_use_mode: bool = True,
        scheduled_sampling_stop_gradient: bool = True,
        return_sampling_diagnostics: bool = False,
    ):
        keys = jax.random.split(seed, 2 * self.num_subgoals + 1)
        subgoal_sample_keys = keys[: self.num_subgoals]
        mask_keys = keys[self.num_subgoals : 2 * self.num_subgoals]
        action_sample_key = keys[-1]

        observations = jnp.expand_dims(observations, axis=1)
        if goals is not None:
            goals = jnp.expand_dims(goals, axis=1)

        if self.gc_encoder is not None:
            features = self.gc_encoder(observations, goals, goal_encoded=False, listwise=True)
            obs_feature = self.feature_embed(features[0])
            goal_feature = features[1]
            features = jnp.concatenate([obs_feature, goal_feature], axis=1)
        else:
            features = [self.feature_embed(observations)]
            if goals is not None:
                features.append(self.feature_embed(goals))
            features = jnp.concatenate(features, axis=1)

        batch_size = features.shape[0]
        predicted_subgoals = jnp.zeros((batch_size, self.num_subgoals, self.joint_embed_dim), dtype=jnp.float32)
        history_subgoals = jnp.zeros_like(predicted_subgoals)
        prev_embed_tokens = jnp.tile(self.prev_tokens, (batch_size, 1, 1))

        high_dist_list = []
        diagnostics_list = []
        sampling_diagnostics = {} if return_sampling_diagnostics else None
        sampling_ratios = []
        for token_dim in range(self.num_subgoals + 1):
            if token_dim == 0:
                prev_embeds = prev_embed_tokens
            else:
                prev_embeds = jnp.concatenate(
                    [history_subgoals[:, :token_dim, :], prev_embed_tokens[:, token_dim:, :]],
                    axis=1,
                )

            x = jnp.concatenate([features, prev_embeds], axis=1)
            target_dim = features.shape[1] + token_dim + 1
            if return_diagnostics:
                x, diagnostics = self.hcoghp_mix(x, return_diagnostics=True)
                diagnostics_list.append(diagnostics)
            else:
                x = self.hcoghp_mix(x)
            target_token = x[:, target_dim - 1, :]

            if token_dim < self.num_subgoals:
                high_dist = self.high_actor_head(target_token, temperature=temperature)
                high_dist_list.append(high_dist)

                if subgoal_reps is not None and scheduled_sampling_use_mode:
                    predicted_token = high_dist.mode()
                else:
                    predicted_token = high_dist.sample(seed=subgoal_sample_keys[token_dim])

                predicted_subgoals = predicted_subgoals.at[:, token_dim, :].set(predicted_token)

                if subgoal_reps is None:
                    history_token = predicted_token
                else:
                    predicted_history_token = predicted_token
                    if scheduled_sampling_stop_gradient:
                        predicted_history_token = jax.lax.stop_gradient(predicted_history_token)
                    mask = jax.random.bernoulli(
                        mask_keys[token_dim],
                        p=scheduled_sampling_prob,
                        shape=(batch_size, 1),
                    )
                    history_token = jnp.where(mask, predicted_history_token, subgoal_reps[:, token_dim, :])
                    if return_sampling_diagnostics:
                        ratio = jnp.mean(mask.astype(jnp.float32))
                        sampling_diagnostics[f'scheduled_sampling_ratio_{token_dim}'] = ratio
                        sampling_ratios.append(ratio)

                history_subgoals = history_subgoals.at[:, token_dim, :].set(history_token)
            else:
                low_dist = self.low_actor_head(target_token, temperature=temperature)
                predicted_actions = low_dist.sample(seed=action_sample_key)

        if return_sampling_diagnostics:
            sampling_diagnostics['scheduled_sampling_prob'] = jnp.asarray(scheduled_sampling_prob, dtype=jnp.float32)
            if sampling_ratios:
                sampling_diagnostics['scheduled_sampling_ratio'] = jnp.mean(jnp.stack(sampling_ratios))
            elif subgoal_reps is None:
                sampling_diagnostics['scheduled_sampling_ratio'] = jnp.array(1.0)
            else:
                sampling_diagnostics['scheduled_sampling_ratio'] = jnp.array(0.0)

        if return_diagnostics or return_sampling_diagnostics:
            diagnostics = {}
            if return_diagnostics:
                diagnostics.update(self.average_diagnostics(diagnostics_list))
            if return_sampling_diagnostics:
                diagnostics.update(sampling_diagnostics)
            return high_dist_list, low_dist, predicted_actions, diagnostics

        return high_dist_list, low_dist, predicted_actions
