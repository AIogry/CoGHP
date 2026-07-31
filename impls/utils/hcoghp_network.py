"""HCoGHP actor: CoGHP autoregression with HRM subgoal-state updates."""

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.coghp_network import MixerBlock
from utils.networks import LengthNormalize, MLP, default_init


def apply_stage_gradient_boundary(
    z_low: jnp.ndarray,
    z_high: jnp.ndarray,
    selected_subgoal_token: jnp.ndarray,
    gradient_mode: str,
):
    """Apply one consistent backward boundary without changing forward values."""
    if gradient_mode == 'stage_truncated':
        return (
            jax.lax.stop_gradient(z_low),
            jax.lax.stop_gradient(z_high),
            jax.lax.stop_gradient(selected_subgoal_token),
        )
    if gradient_mode == 'full_bptt':
        return z_low, z_high, selected_subgoal_token
    raise ValueError(
        f'Unknown hrm_gradient_mode {gradient_mode!r}; expected '
        "'stage_truncated' or 'full_bptt'."
    )


class HRMLowMixerCell(nn.Module):
    """Shared low-state update over fixed-role feature tokens.

    Tokens are ordered as ``[context, stage_embedding, z_high, z_low]``.
    MixerBlock's internal residuals are retained, but no extra recurrent-state
    residual or gate is applied outside the cell.
    """

    state_dim: int
    mixer_token_hidden: int
    mixer_channel_hidden: int
    num_mixer_blocks: int = 1
    layer_norm: bool = True

    def setup(self):
        if self.num_mixer_blocks < 1:
            raise ValueError('HRMLowMixerCell requires at least one MixerBlock.')
        mixer_kwargs = dict(
            num_tokens=4,
            embed_dim=self.state_dim,
            hidden_dim_tokens=self.mixer_token_hidden,
            hidden_dim_channels=self.mixer_channel_hidden,
            causal=False,
        )
        self.mixer_blocks = [
            MixerBlock(**mixer_kwargs) for _ in range(self.num_mixer_blocks)
        ]
        if self.layer_norm:
            self.output_norm = nn.LayerNorm()

    def __call__(
        self,
        z_low: jnp.ndarray,
        z_high: jnp.ndarray,
        context: jnp.ndarray,
        stage_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        tokens = jnp.stack([context, stage_embedding, z_high, z_low], axis=1)
        for mixer_block in self.mixer_blocks:
            tokens = mixer_block(tokens)
        new_z_low = tokens[:, 3, :]
        if self.layer_norm:
            new_z_low = self.output_norm(new_z_low)
        return new_z_low


class HRMHighMixerCell(nn.Module):
    """Shared high-state update over fixed-role feature tokens.

    Tokens are ordered as ``[context, stage_embedding, z_low, z_high]``.
    The returned value directly replaces the previous high state.
    """

    state_dim: int
    mixer_token_hidden: int
    mixer_channel_hidden: int
    num_mixer_blocks: int = 1
    layer_norm: bool = True

    def setup(self):
        if self.num_mixer_blocks < 1:
            raise ValueError('HRMHighMixerCell requires at least one MixerBlock.')
        mixer_kwargs = dict(
            num_tokens=4,
            embed_dim=self.state_dim,
            hidden_dim_tokens=self.mixer_token_hidden,
            hidden_dim_channels=self.mixer_channel_hidden,
            causal=False,
        )
        self.mixer_blocks = [
            MixerBlock(**mixer_kwargs) for _ in range(self.num_mixer_blocks)
        ]
        if self.layer_norm:
            self.output_norm = nn.LayerNorm()

    def __call__(
        self,
        z_high: jnp.ndarray,
        z_low: jnp.ndarray,
        context: jnp.ndarray,
        stage_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        tokens = jnp.stack([context, stage_embedding, z_low, z_high], axis=1)
        for mixer_block in self.mixer_blocks:
            tokens = mixer_block(tokens)
        new_z_high = tokens[:, 3, :]
        if self.layer_norm:
            new_z_high = self.output_norm(new_z_high)
        return new_z_high


class SubgoalReadout(nn.Module):
    """Map recurrent state and context to a subgoal actor representation."""

    state_dim: int
    hidden_dim: int
    layer_norm: bool = True

    def setup(self):
        self.hidden_dense = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.output_dense = nn.Dense(self.state_dim, kernel_init=default_init())
        if self.layer_norm:
            self.hidden_norm = nn.LayerNorm()
            self.output_norm = nn.LayerNorm()

    def __call__(
        self,
        context: jnp.ndarray,
        z_low: jnp.ndarray,
        z_high: jnp.ndarray,
        stage_embedding: jnp.ndarray,
    ) -> jnp.ndarray:
        readout_input = jnp.concatenate(
            [context, z_low, z_high, stage_embedding],
            axis=-1,
        )
        hidden = nn.gelu(self.hidden_dense(readout_input))
        if self.layer_norm:
            hidden = self.hidden_norm(hidden)
        stage_rep = self.output_dense(hidden)
        if self.layer_norm:
            stage_rep = self.output_norm(stage_rep)
        return stage_rep


class HCoGHPPolicyNetwork(nn.Module):
    """CoGHP outer autoregression with HRM only for latent subgoals.

    A shared context Mixer stack reads each current query from the full CoGHP
    token sequence. During the K subgoal stages, shared low/high HRM cells
    update feature-level hidden states that both carry continuously between
    stages. A shared SubgoalReadout maps context and hidden states to the
    high-actor input.

    The final action stage reuses only the context Mixer. It reads the action
    query after the selected subgoal chain has been written into the prefix and
    passes that context directly to the original low actor head. Therefore the
    action stage neither invokes nor updates the HRM hidden states.
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
    causal_mixer: bool = True
    action_use_full_subgoal_chain: bool = True
    hrm_l_cycles: int = 2
    hrm_init_std: float = 0.1
    hrm_trainable_init: bool = False
    hrm_low_num_blocks: int = 1
    hrm_high_num_blocks: int = 1
    subgoal_readout_hidden_dim: int = 256
    enable_hcoghp_diagnostics: bool = False
    hrm_gradient_mode: str = 'stage_truncated'

    def setup(self):
        if self.num_mixer_blocks < 1:
            raise ValueError('num_mixer_blocks must be at least 1.')
        if self.hrm_l_cycles < 1:
            raise ValueError('hrm_l_cycles must be at least 1.')
        if self.hrm_low_num_blocks < 1:
            raise ValueError('hrm_low_num_blocks must be at least 1.')
        if self.hrm_high_num_blocks < 1:
            raise ValueError('hrm_high_num_blocks must be at least 1.')
        if self.subgoal_readout_hidden_dim < 1:
            raise ValueError('subgoal_readout_hidden_dim must be at least 1.')
        if self.hrm_gradient_mode not in ('stage_truncated', 'full_bptt'):
            raise ValueError(
                f'Unknown hrm_gradient_mode {self.hrm_gradient_mode!r}; expected '
                "'stage_truncated' or 'full_bptt'."
            )

        self.prev_tokens = self.param(
            'prev_tokens',
            nn.initializers.normal(stddev=0.1),
            (1, self.num_subgoals + 1, self.state_dim),
        )
        self.z_l_init = self.param(
            'z_l_init',
            nn.initializers.normal(stddev=self.hrm_init_std),
            (1, self.state_dim),
        )
        self.z_h_init = self.param(
            'z_h_init',
            nn.initializers.normal(stddev=self.hrm_init_std),
            (1, self.state_dim),
        )

        context_mixer_kwargs = dict(
            num_tokens=self.num_subgoals + 3,
            embed_dim=self.state_dim,
            hidden_dim_tokens=self.mixer_token_hidden,
            hidden_dim_channels=self.mixer_channel_hidden,
            causal=self.causal_mixer,
        )
        # Different depth blocks have independent parameters. This whole stack
        # is reused at every subgoal stage and at the action stage.
        self.context_mixer_blocks = [
            MixerBlock(**context_mixer_kwargs) for _ in range(self.num_mixer_blocks)
        ]
        self.stage_embedding = nn.Embed(
            num_embeddings=self.num_subgoals + 1,
            features=self.state_dim,
        )
        self.low_cell = HRMLowMixerCell(
            state_dim=self.state_dim,
            mixer_token_hidden=self.mixer_token_hidden,
            mixer_channel_hidden=self.mixer_channel_hidden,
            num_mixer_blocks=self.hrm_low_num_blocks,
            layer_norm=self.layer_norm,
        )
        self.high_cell = HRMHighMixerCell(
            state_dim=self.state_dim,
            mixer_token_hidden=self.mixer_token_hidden,
            mixer_channel_hidden=self.mixer_channel_hidden,
            num_mixer_blocks=self.hrm_high_num_blocks,
            layer_norm=self.layer_norm,
        )
        self.subgoal_readout = SubgoalReadout(
            state_dim=self.state_dim,
            hidden_dim=self.subgoal_readout_hidden_dim,
            layer_norm=self.layer_norm,
        )
        self.feature_embed = nn.Sequential(
            [
                MLP(
                    hidden_dims=(*self.enc_hidden, self.state_dim),
                    activate_final=False,
                    layer_norm=True,
                ),
                LengthNormalize(),
            ]
        )

    @staticmethod
    def _mean_norm(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.linalg.norm(x, axis=-1))

    @staticmethod
    def _mean_relative_change(
        new: jnp.ndarray,
        old: jnp.ndarray,
        eps: float = 1e-8,
    ) -> jnp.ndarray:
        numerator = jnp.linalg.norm(new - old, axis=-1)
        denominator = jnp.linalg.norm(old, axis=-1) + eps
        return jnp.mean(numerator / denominator)

    @staticmethod
    def _mean_cosine(
        left: jnp.ndarray,
        right: jnp.ndarray,
        eps: float = 1e-8,
    ) -> jnp.ndarray:
        numerator = jnp.sum(left * right, axis=-1)
        denominator = (
            jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1) + eps
        )
        return jnp.mean(numerator / denominator)

    def _initial_state(self, state: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        if not self.hrm_trainable_init:
            state = jax.lax.stop_gradient(state)
        return jnp.tile(state, (batch_size, 1))

    def _encode_inputs(
        self,
        observations: jnp.ndarray,
        goals: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        observations = jnp.expand_dims(observations, axis=1)
        if goals is not None:
            goals = jnp.expand_dims(goals, axis=1)

        if self.gc_encoder is not None:
            features = self.gc_encoder(
                observations,
                goals,
                goal_encoded=False,
                listwise=True,
            )
            obs_feature = self.feature_embed(features[0])
            goal_feature = features[1]
            return jnp.concatenate([obs_feature, goal_feature], axis=1)

        features = [self.feature_embed(observations)]
        if goals is not None:
            features.append(self.feature_embed(goals))
        return jnp.concatenate(features, axis=1)

    def _stage_embedding(
        self,
        stage_idx: int,
        batch_size: int,
    ) -> jnp.ndarray:
        stage_ids = jnp.full((batch_size,), stage_idx, dtype=jnp.int32)
        return self.stage_embedding(stage_ids)

    def _context_from_query(
        self,
        token_sequence: jnp.ndarray,
        query_index: int,
        stage_emb: jnp.ndarray,
    ) -> jnp.ndarray:
        # The stage identity modifies only the currently-read query token.
        stage_tokens = token_sequence.at[:, query_index, :].add(stage_emb)
        for context_mixer_block in self.context_mixer_blocks:
            stage_tokens = context_mixer_block(stage_tokens)
        return stage_tokens[:, query_index, :]

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
        """Generate K subgoal distributions followed by one action distribution."""
        del action_seq

        keys = jax.random.split(seed, 2 * self.num_subgoals + 1)
        subgoal_sample_keys = keys[: self.num_subgoals]
        mask_keys = keys[self.num_subgoals : 2 * self.num_subgoals]
        action_sample_key = keys[-1]

        features = self._encode_inputs(observations, goals)
        batch_size = features.shape[0]
        history_subgoals = jnp.zeros(
            (batch_size, self.num_subgoals, self.joint_embed_dim),
            dtype=jnp.float32,
        )
        prev_embed_tokens = jnp.tile(self.prev_tokens, (batch_size, 1, 1))

        # Both states are initialized exactly once and carry only across the K
        # subgoal stages. Teacher forcing changes token history, not this carry.
        z_low = self._initial_state(self.z_l_init, batch_size)
        z_high = self._initial_state(self.z_h_init, batch_size)

        high_dist_list = []
        diagnostics = {} if (return_diagnostics or return_sampling_diagnostics) else None
        sampling_ratios = []
        context_norms = []
        z_low_changes = []
        z_high_changes = []
        readout_norms = []

        for subgoal_idx in range(self.num_subgoals):
            if subgoal_idx == 0:
                prev_embeds = prev_embed_tokens
            else:
                prev_embeds = jnp.concatenate(
                    [
                        history_subgoals[:, :subgoal_idx, :],
                        prev_embed_tokens[:, subgoal_idx:, :],
                    ],
                    axis=1,
                )

            token_sequence = jnp.concatenate([features, prev_embeds], axis=1)
            query_index = features.shape[1] + subgoal_idx
            stage_emb = self._stage_embedding(subgoal_idx, batch_size)
            context = self._context_from_query(
                token_sequence,
                query_index,
                stage_emb,
            )

            z_low_before = z_low
            z_high_before = z_high
            z_low_stage = z_low
            low_update_values = []
            for _ in range(self.hrm_l_cycles):
                old_z_low = z_low_stage
                z_low_stage = self.low_cell(
                    z_low=z_low_stage,
                    z_high=z_high_before,
                    context=context,
                    stage_embedding=stage_emb,
                )
                low_update_values.append(
                    self._mean_relative_change(z_low_stage, old_z_low)
                )

            z_high_stage = self.high_cell(
                z_high=z_high_before,
                z_low=z_low_stage,
                context=context,
                stage_embedding=stage_emb,
            )
            stage_rep = self.subgoal_readout(
                context=context,
                z_low=z_low_stage,
                z_high=z_high_stage,
                stage_embedding=stage_emb,
            )
            high_dist = self.high_actor_head(stage_rep, temperature=temperature)
            high_dist_list.append(high_dist)

            if subgoal_reps is not None and scheduled_sampling_use_mode:
                predicted_token = high_dist.mode()
            else:
                predicted_token = high_dist.sample(
                    seed=subgoal_sample_keys[subgoal_idx]
                )

            if subgoal_reps is None:
                selected_subgoal_token = predicted_token
            else:
                predicted_history_token = predicted_token
                # The unified stage boundary below supersedes this legacy flag
                # in stage_truncated mode. full_bptt deliberately keeps the
                # prediction branch differentiable for the ablation.
                mask = jax.random.bernoulli(
                    mask_keys[subgoal_idx],
                    p=scheduled_sampling_prob,
                    shape=(batch_size, 1),
                )
                selected_subgoal_token = jnp.where(
                    mask,
                    predicted_history_token,
                    subgoal_reps[:, subgoal_idx, :],
                )
                if return_sampling_diagnostics:
                    ratio = jnp.mean(mask.astype(jnp.float32))
                    diagnostics[
                        f'scheduled_sampling_ratio_{subgoal_idx}'
                    ] = ratio
                    sampling_ratios.append(ratio)

            # Current-stage AWR sees the untruncated high_dist above. Only the
            # recurrent carry and selected token entering later stages are
            # detached. Thus AWR defines the planner objective, while this
            # boundary defines whether its graph crosses stage boundaries.
            z_low, z_high, history_token = apply_stage_gradient_boundary(
                z_low_stage,
                z_high_stage,
                selected_subgoal_token,
                self.hrm_gradient_mode,
            )
            history_subgoals = history_subgoals.at[:, subgoal_idx, :].set(
                history_token
            )

            context_norm = self._mean_norm(context)
            z_low_change = self._mean_relative_change(
                z_low_stage,
                z_low_before,
            )
            z_high_change = self._mean_relative_change(
                z_high_stage,
                z_high_before,
            )
            readout_norm = self._mean_norm(stage_rep)
            context_norms.append(context_norm)
            z_low_changes.append(z_low_change)
            z_high_changes.append(z_high_change)
            readout_norms.append(readout_norm)

            if return_diagnostics:
                prefix = f'stage_{subgoal_idx}'
                diagnostics[f'{prefix}/context_norm'] = context_norm
                diagnostics[f'{prefix}/z_low_before_norm'] = self._mean_norm(
                    z_low_before
                )
                diagnostics[f'{prefix}/z_low_after_norm'] = self._mean_norm(
                    z_low_stage
                )
                diagnostics[f'{prefix}/z_low_relative_change'] = z_low_change
                diagnostics[f'{prefix}/z_high_before_norm'] = self._mean_norm(
                    z_high_before
                )
                diagnostics[f'{prefix}/z_high_after_norm'] = self._mean_norm(
                    z_high_stage
                )
                diagnostics[f'{prefix}/z_high_relative_change'] = z_high_change
                diagnostics[f'{prefix}/z_low_high_cosine'] = self._mean_cosine(
                    z_low_stage,
                    z_high_stage,
                )
                diagnostics[f'{prefix}/subgoal_readout_norm'] = readout_norm
                diagnostics[f'{prefix}/stage_embedding_norm'] = self._mean_norm(
                    stage_emb
                )
                for low_idx, low_update in enumerate(low_update_values):
                    diagnostics[f'{prefix}/low_update_{low_idx}'] = low_update
                diagnostics[f'{prefix}/carry_truncated'] = jnp.asarray(
                    self.hrm_gradient_mode == 'stage_truncated',
                    dtype=jnp.float32,
                )

        # The action uses the completed autoregressive token prefix and the same
        # context Mixer, but deliberately does not invoke or update HRM state.
        if self.num_subgoals == 0:
            action_prev_embeds = prev_embed_tokens
        else:
            action_history = history_subgoals
            if not self.action_use_full_subgoal_chain:
                action_history = prev_embed_tokens[
                    :, : self.num_subgoals, :
                ].at[:, self.num_subgoals - 1, :].set(
                    history_subgoals[:, self.num_subgoals - 1, :]
                )
            action_prev_embeds = jnp.concatenate(
                [
                    action_history,
                    prev_embed_tokens[:, self.num_subgoals :, :],
                ],
                axis=1,
            )

        action_token_sequence = jnp.concatenate(
            [features, action_prev_embeds],
            axis=1,
        )
        action_query_index = features.shape[1] + self.num_subgoals
        action_stage_emb = self._stage_embedding(self.num_subgoals, batch_size)
        action_context = self._context_from_query(
            action_token_sequence,
            action_query_index,
            action_stage_emb,
        )
        low_dist = self.low_actor_head(action_context, temperature=temperature)
        predicted_actions = low_dist.sample(seed=action_sample_key)

        if return_diagnostics:
            diagnostics['action/context_norm'] = self._mean_norm(action_context)
            diagnostics['action/stage_embedding_norm'] = self._mean_norm(
                action_stage_emb
            )
            diagnostics['action/action_mode_norm'] = self._mean_norm(low_dist.mode())
            diagnostics['hrm/num_stages'] = jnp.asarray(
                self.num_subgoals,
                dtype=jnp.float32,
            )
            diagnostics['hrm/gradient_mode_is_stage_truncated'] = jnp.asarray(
                self.hrm_gradient_mode == 'stage_truncated',
                dtype=jnp.float32,
            )
            if self.num_subgoals > 0:
                diagnostics['hrm/context_norm'] = jnp.mean(jnp.stack(context_norms))
                diagnostics['hrm/z_low_final_norm'] = self._mean_norm(z_low)
                diagnostics['hrm/z_high_final_norm'] = self._mean_norm(z_high)
                diagnostics['hrm/z_low_stage_change'] = jnp.mean(
                    jnp.stack(z_low_changes)
                )
                diagnostics['hrm/z_high_stage_change'] = jnp.mean(
                    jnp.stack(z_high_changes)
                )
                diagnostics['hrm/subgoal_readout_norm'] = jnp.mean(
                    jnp.stack(readout_norms)
                )

        if return_sampling_diagnostics:
            diagnostics['scheduled_sampling_prob'] = jnp.asarray(
                scheduled_sampling_prob,
                dtype=jnp.float32,
            )
            if sampling_ratios:
                diagnostics['scheduled_sampling_ratio'] = jnp.mean(
                    jnp.stack(sampling_ratios)
                )
            elif subgoal_reps is None:
                diagnostics['scheduled_sampling_ratio'] = jnp.array(1.0)
            else:
                diagnostics['scheduled_sampling_ratio'] = jnp.array(0.0)

        if return_diagnostics or return_sampling_diagnostics:
            return high_dist_list, low_dist, predicted_actions, diagnostics
        return high_dist_list, low_dist, predicted_actions
