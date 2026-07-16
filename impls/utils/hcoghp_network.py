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

    def hcoghp_mix(self, x_base):
        z_low = x_base
        z_high = x_base

        for _ in range(self.num_mixer_blocks):
            for _ in range(self.hrm_l_cycles):
                z_low = self.low_mixer_block(self.mix_states(x_base, z_low, z_high))
            z_high = self.high_mixer_block(self.mix_states(x_base, z_low, z_high))

        return z_high

    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: int = None,
        subgoal_reps: Optional[jnp.ndarray] = None,
        action_seq: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
    ):
        high_seed, low_seed = jax.random.split(seed)

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
        prev_embed_tokens = jnp.tile(self.prev_tokens, (batch_size, 1, 1))

        high_dist_list = []
        for token_dim in range(self.num_subgoals + 1):
            if token_dim == 0:
                prev_embeds = prev_embed_tokens
            elif subgoal_reps is not None:
                prev_embeds = jnp.concatenate(
                    [subgoal_reps[:, :token_dim, :], prev_embed_tokens[:, token_dim:, :]],
                    axis=1,
                )
            else:
                prev_embeds = jnp.concatenate(
                    [predicted_subgoals[:, :token_dim, :], prev_embed_tokens[:, token_dim:, :]],
                    axis=1,
                )

            x = jnp.concatenate([features, prev_embeds], axis=1)
            target_dim = features.shape[1] + token_dim + 1
            x = self.hcoghp_mix(x)
            target_token = x[:, target_dim - 1, :]

            if token_dim < self.num_subgoals:
                high_dist = self.high_actor_head(target_token, temperature=temperature)
                high_dist_list.append(high_dist)
                goal_reps = high_dist.sample(seed=high_seed)
                predicted_subgoals = predicted_subgoals.at[:, token_dim, :].set(goal_reps)
            else:
                low_dist = self.low_actor_head(target_token, temperature=temperature)
                predicted_actions = low_dist.sample(seed=low_seed)

        return high_dist_list, low_dist, predicted_actions
