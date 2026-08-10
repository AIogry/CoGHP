"""CoGHP planner with configurable shared-block primitive-action cores.

The planner preserves CoGHP's high-level far-to-near autoregression.  The
primitive-action branch builds the complete ``[observation, goal, subgoal
chain, action query]`` token sequence, applies the original stateless CoGHP
Mixer or one of the recurrent/shared-block action-core schedules, and reads
the final action-query token.

There are no segments, stage embeddings, residual action corrections, custom
H/L cells, compressed context vectors, or cross-environment-step carries in
this implementation.
"""

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.coghp_network import MixerBlock
from utils.hrm import ACTION_CORE_TYPES, HRM_BLOCK_TYPES, HRMCore
from utils.networks import LengthNormalize, MLP


_FORWARD_DTYPES = {
    'float32': jnp.float32,
    'bfloat16': jnp.bfloat16,
}

_ACTION_CORE_TYPES = ACTION_CORE_TYPES | frozenset({'vanilla_coghp'})


def resolve_action_block_type(
    action_block_type: str,
    legacy_hrm_block_type: str,
    legacy_is_mlp_mixer: bool,
) -> str:
    """Resolve canonical and legacy block selectors deterministically.

    ``attention`` remains accepted as an explicit, HRM-only compatibility
    value.  It is deliberately not part of :data:`HRM_BLOCK_TYPES` because
    the three public action Mixer families are the values in that set.
    """

    if action_block_type not in (None, '', 'auto', 'legacy'):
        if action_block_type == 'attention':
            return action_block_type
        if action_block_type not in HRM_BLOCK_TYPES:
            raise ValueError(
                f'Unknown action_block_type {action_block_type!r}; expected '
                f'one of {sorted(HRM_BLOCK_TYPES)}, auto, or the legacy '
                'HRM-only attention value.'
            )
        return action_block_type

    if legacy_hrm_block_type not in (None, '', 'auto', 'legacy'):
        if legacy_hrm_block_type not in HRM_BLOCK_TYPES:
            raise ValueError(
                f'Unknown legacy hrm_block_type '
                f'{legacy_hrm_block_type!r}; expected one of '
                f'{sorted(HRM_BLOCK_TYPES)} or legacy.'
            )
        return legacy_hrm_block_type

    return 'swiglu_rmsnorm' if legacy_is_mlp_mixer else 'attention'


class HRMActionCoGHPPolicyNetwork(nn.Module):
    """CoGHP planner plus a configurable full-sequence action core."""

    state_dim: int
    high_actor_head: nn.Module
    low_actor_head: nn.Module

    num_mixer_blocks: int = 1
    mixer_token_hidden: int = 64
    mixer_channel_hidden: int = 64
    gc_encoder: Optional[nn.Module] = None
    enc_hidden: Sequence[int] = (128, 128)
    num_subgoals: int = 1
    causal_mixer: bool = True

    planner_core_type: str = 'vanilla_coghp'        # vanilla CoGHP, matched feed-forward, or official HRM
    planner_block_type: str = 'swiglu_rmsnorm'
    planner_num_layers: int = 1
    planner_mixer_intermediate_size: int = 256      # swiglu rmsnorm hidden size
    planner_h_cycles: int = 2
    planner_l_cycles: int = 2
    planner_bptt: bool = True
    planner_norm_eps: float = 1e-6
    planner_forward_dtype: str = 'float32'

    # When enabled, the planner and primitive-action branches use one
    # official HRM parameter tree.  The two CoGHP readouts remain separate:
    # the high readout consumes planner query tokens and the low readout
    # consumes the final action-query token.  Each invocation still starts
    # from a fresh transient carry; no recurrent state crosses a subgoal or
    # environment step.
    share_actor_core: bool = False

    action_hrm_num_layers: int = 1
    action_hrm_intermediate_size: int = 512
    action_hrm_head_dim: int = 32
    action_hrm_h_cycles: int = 2
    action_hrm_l_cycles: int = 2
    action_hrm_bptt: bool = True
    action_hrm_is_causal: bool = True
    action_hrm_is_mlp_mixer: bool = True
    action_hrm_mlp_mixer_intermediate_size: int = 256
    action_core_type: str = 'hrm'
    action_block_type: str = 'auto'
    hrm_block_type: str = 'legacy'
    hrm_token_hidden: int = 32
    hrm_channel_hidden: int = 32
    hrm_mixer_intermediate_size: int = -1
    hrm_layernorm_affine: bool = True
    action_hrm_norm_eps: float = 1e-6
    action_hrm_rope_theta: float = 10000.0
    action_hrm_forward_dtype: str = 'float32'

    train_action_chain_source: str = 'teacher'

    def setup(self):
        if self.state_dim < 1:
            raise ValueError('state_dim must be positive.')
        if self.num_subgoals < 1:
            raise ValueError('HRM-Action-CoGHP requires num_subgoals >= 1.')
        if self.num_mixer_blocks < 1:
            raise ValueError('num_mixer_blocks must be positive.')
        if self.planner_core_type not in (
            'vanilla_coghp',
            'legacy_coghp',
            'matched_feedforward',
            'hrm',
        ):
            raise ValueError(
                'planner_core_type must be vanilla_coghp (or its legacy '
                'alias legacy_coghp), '
                'matched_feedforward, or hrm; '
                f'got {self.planner_core_type!r}.'
            )

        if self.planner_num_layers < 1:
            raise ValueError('planner_num_layers must be positive.')

        if self.planner_mixer_intermediate_size < 1:
            raise ValueError(
                'planner_mixer_intermediate_size must be positive.'
            )

        if self.planner_h_cycles < 1 or self.planner_l_cycles < 1:
            raise ValueError(
                'planner_h_cycles and planner_l_cycles must be positive.'
            )
        if not isinstance(self.planner_bptt, bool):
            raise ValueError('planner_bptt must be a bool.')
        if not isinstance(self.share_actor_core, bool):
            raise ValueError('share_actor_core must be a bool.')

        if self.planner_forward_dtype not in _FORWARD_DTYPES:
            raise ValueError(
                'planner_forward_dtype must be float32 or bfloat16; '
                f'got {self.planner_forward_dtype!r}.'
            )

        if (
            self.planner_core_type in ('matched_feedforward', 'hrm')
            and self.planner_block_type != 'swiglu_rmsnorm'
        ):
            raise ValueError(
                'The non-vanilla planner currently supports only '
                'planner_block_type=swiglu_rmsnorm.'
            )
        if self.share_actor_core and (
            self.planner_core_type != 'hrm'
            or self.action_core_type != 'hrm'
        ):
            raise ValueError(
                'share_actor_core requires planner_core_type=hrm and '
                'action_core_type=hrm.'
            )
        if self.share_actor_core and self.planner_block_type != 'swiglu_rmsnorm':
            raise ValueError(
                'share_actor_core currently requires '
                'planner_block_type=swiglu_rmsnorm.'
            )
        if self.action_hrm_num_layers < 1:
            raise ValueError('action_hrm_num_layers must be positive.')
        if self.action_hrm_h_cycles < 1 or self.action_hrm_l_cycles < 1:
            raise ValueError('HRM H_cycles and L_cycles must be positive.')
        if self.action_hrm_intermediate_size < 1:
            raise ValueError('action_hrm_intermediate_size must be positive.')
        if self.action_core_type not in _ACTION_CORE_TYPES:
            raise ValueError(
                f'action_core_type must be one of '
                f'{sorted(_ACTION_CORE_TYPES)}; got '
                f'{self.action_core_type!r}.'
            )
        if self.hrm_mixer_intermediate_size == 0 or (
            self.hrm_mixer_intermediate_size < -1
        ):
            raise ValueError(
                'hrm_mixer_intermediate_size must be positive or -1 for '
                'the legacy action_hrm_mlp_mixer_intermediate_size alias.'
            )
        resolved_block_type = resolve_action_block_type(
            self.action_block_type,
            self.hrm_block_type,
            self.action_hrm_is_mlp_mixer,
        )
        if (
            self.action_core_type == 'vanilla_coghp'
            and resolved_block_type == 'attention'
        ):
            raise ValueError(
                'vanilla_coghp executor requires the CoGHP Mixer path; '
                'set action_hrm_backbone=mixer.'
            )
        if (
            self.action_core_type != 'hrm'
            and resolved_block_type == 'attention'
        ):
            raise ValueError(
                'matched_feedforward and single_state_iterative support only '
                f'action_block_type values {sorted(HRM_BLOCK_TYPES)}.'
            )
        if (
            resolved_block_type == 'swiglu_rmsnorm'
            and self.hrm_mixer_intermediate_size == -1
            and self.action_hrm_mlp_mixer_intermediate_size < 1
        ):
            raise ValueError(
                'action_hrm_mlp_mixer_intermediate_size must be positive '
                'when hrm_mixer_intermediate_size uses its -1 alias.'
            )
        if resolved_block_type in ('coghp_gelu', 'gelu_layernorm') and (
            self.hrm_token_hidden < 1 or self.hrm_channel_hidden < 1
        ):
            raise ValueError(
                'hrm_token_hidden and hrm_channel_hidden must be positive '
                f'for {resolved_block_type}.'
            )
        if self.train_action_chain_source not in ('teacher', 'predicted'):
            raise ValueError(
                'train_action_chain_source must be teacher or predicted; got '
                f'{self.train_action_chain_source!r}.'
            )
        if self.action_hrm_forward_dtype not in _FORWARD_DTYPES:
            raise ValueError(
                'action_hrm_forward_dtype must be float32 or bfloat16; got '
                f'{self.action_hrm_forward_dtype!r}.'
            )
        if resolved_block_type == 'attention':
            if (
                self.action_hrm_head_dim < 1
                or self.state_dim % self.action_hrm_head_dim != 0
            ):
                raise ValueError(
                    'Attention HRM requires action_hrm_head_dim to divide '
                    'state_dim.'
                )

        if self.share_actor_core and resolved_block_type != 'swiglu_rmsnorm':
            raise ValueError(
                'share_actor_core currently requires '
                'action_block_type=swiglu_rmsnorm.'
            )
        if self.share_actor_core and (
            self.planner_num_layers != self.action_hrm_num_layers
            or self.planner_mixer_intermediate_size
            != (
                self.hrm_mixer_intermediate_size
                if self.hrm_mixer_intermediate_size != -1
                else self.action_hrm_mlp_mixer_intermediate_size
            )
            or self.planner_forward_dtype != self.action_hrm_forward_dtype
            or self.planner_norm_eps != self.action_hrm_norm_eps
            or self.action_hrm_is_causal != self.causal_mixer
        ):
            raise ValueError(
                'share_actor_core requires matching planner/action HRM '
                'num_layers, mixer_intermediate_size, dtype, norm, and '
                'causal settings.'
            )

        # The K+1 planner placeholders reproduce the original CoGHP planner
        # sequence.  The action branch owns a distinct query parameter.
        self.planner_query_tokens = self.param(
            'planner_query_tokens',
            nn.initializers.normal(stddev=0.1),
            (1, self.num_subgoals + 1, self.state_dim),
        )
        self.action_query_token = self.param(
            'action_query_token',
            nn.initializers.normal(stddev=0.1),
            (1, 1, self.state_dim),
        )

        sequence_length = self.num_subgoals + 3

        if self.share_actor_core:
            shared_forward_dtype = _FORWARD_DTYPES[
                self.planner_forward_dtype
            ]
            self.shared_actor_core = HRMCore(
                seq_len=sequence_length,
                num_layers=self.planner_num_layers,
                hidden_size=self.state_dim,
                intermediate_size=self.action_hrm_intermediate_size,
                head_dim=self.action_hrm_head_dim,
                is_causal=self.causal_mixer,
                h_cycles=self.planner_h_cycles,
                l_cycles=self.planner_l_cycles,
                core_type='hrm',
                bptt=self.planner_bptt,
                norm_eps=self.planner_norm_eps,
                rope_theta=self.action_hrm_rope_theta,
                is_mlp_mixer=True,
                mlp_mixer_intermediate_size=self.planner_mixer_intermediate_size,
                block_type='swiglu_rmsnorm',
                token_hidden=self.hrm_token_hidden,
                channel_hidden=self.hrm_channel_hidden,
                layernorm_affine=self.hrm_layernorm_affine,
                forward_dtype=shared_forward_dtype,
                name='shared_actor_core',
            )
        elif self.planner_core_type in ('vanilla_coghp', 'legacy_coghp'):
            planner_mixer_kwargs = dict(
                num_tokens=sequence_length,
                embed_dim=self.state_dim,
                hidden_dim_tokens=self.mixer_token_hidden,
                hidden_dim_channels=self.mixer_channel_hidden,
                causal=self.causal_mixer,
            )
            self.planner_mixer_blocks = [
                MixerBlock(**planner_mixer_kwargs, name=f'planner_mixer_{idx}')         # using the MLP-mixer from coghp_network.py, with GeLU activation and only one forward pass
                for idx in range(self.num_mixer_blocks)
            ]
        else:   # matched_feedforward or official HRM planner
            planner_forward_dtype = _FORWARD_DTYPES[
                self.planner_forward_dtype
            ]

            self.planner_core = HRMCore(
                seq_len=sequence_length,
                num_layers=self.planner_num_layers,
                hidden_size=self.state_dim,

                # These two arguments belong to the generic HRMCore
                # interface. They are not used by the SwiGLU Mixer branch.
                intermediate_size=self.action_hrm_intermediate_size,
                head_dim=self.action_hrm_head_dim,

                is_causal=self.causal_mixer,

                h_cycles=(
                    1
                    if self.planner_core_type == 'matched_feedforward'
                    else self.planner_h_cycles
                ),
                l_cycles=(
                    1
                    if self.planner_core_type == 'matched_feedforward'
                    else self.planner_l_cycles
                ),

                core_type=self.planner_core_type,
                bptt=(
                    True
                    if self.planner_core_type == 'matched_feedforward'
                    else self.planner_bptt
                ),

                norm_eps=self.planner_norm_eps,
                rope_theta=self.action_hrm_rope_theta,

                is_mlp_mixer=True,
                mlp_mixer_intermediate_size=(
                    self.planner_mixer_intermediate_size
                ),
                block_type=self.planner_block_type,

                # Ignored by the SwiGLU block, but required by the
                # shared Transformer interface.
                token_hidden=self.mixer_token_hidden,
                channel_hidden=self.mixer_channel_hidden,
                layernorm_affine=True,

                forward_dtype=planner_forward_dtype,
                name='planner_core',
            )

        if self.share_actor_core:
            pass
        elif self.action_core_type == 'vanilla_coghp':
            # Exact original CoGHP action executor: one or more independent
            # GELU MixerBlock layers over the complete action sequence.  This
            # path deliberately does not create HRM states or an action_hrm
            # module, so it is not confused with matched_feedforward using a
            # CoGHP-GELU block.
            action_mixer_kwargs = dict(
                num_tokens=sequence_length,
                embed_dim=self.state_dim,
                hidden_dim_tokens=self.mixer_token_hidden,
                hidden_dim_channels=self.mixer_channel_hidden,
                causal=self.causal_mixer,
            )
            self.action_mixer_blocks = [
                MixerBlock(
                    **action_mixer_kwargs,
                    name=f'action_mixer_blocks_{idx}',
                )
                for idx in range(self.num_mixer_blocks)
            ]
        else:
            forward_dtype = _FORWARD_DTYPES[self.action_hrm_forward_dtype]
            mixer_intermediate_size = self.hrm_mixer_intermediate_size
            if mixer_intermediate_size == -1:
                mixer_intermediate_size = (
                    self.action_hrm_mlp_mixer_intermediate_size
                )
            self.action_hrm = HRMCore(
                seq_len=sequence_length,
                num_layers=self.action_hrm_num_layers,
                hidden_size=self.state_dim,
                intermediate_size=self.action_hrm_intermediate_size,
                head_dim=self.action_hrm_head_dim,
                is_causal=self.action_hrm_is_causal,
                h_cycles=self.action_hrm_h_cycles,
                l_cycles=self.action_hrm_l_cycles,
                core_type=self.action_core_type,
                bptt=self.action_hrm_bptt,
                norm_eps=self.action_hrm_norm_eps,
                rope_theta=self.action_hrm_rope_theta,
                is_mlp_mixer=(resolved_block_type != 'attention'),
                mlp_mixer_intermediate_size=(
                    mixer_intermediate_size
                ),
                block_type=resolved_block_type,
                token_hidden=self.hrm_token_hidden,
                channel_hidden=self.hrm_channel_hidden,
                layernorm_affine=self.hrm_layernorm_affine,
                forward_dtype=forward_dtype,
                name='action_hrm',
            )

        self.feature_embed = nn.Sequential(
            [
                MLP(
                    hidden_dims=(*self.enc_hidden, self.state_dim),
                    activate_final=False,
                    layer_norm=True,
                ),
                LengthNormalize(),
            ],
            name='feature_embed',
        )

    @staticmethod
    def _mean_norm(value: jax.Array) -> jax.Array:
        return jnp.mean(jnp.linalg.norm(value, axis=-1))

    @staticmethod
    def _mean_relative_change(
        new_value: jax.Array,
        old_value: jax.Array,
        eps: float = 1e-8,
    ) -> jax.Array:
        change = jnp.linalg.norm(new_value - old_value, axis=-1)
        old_norm = jnp.linalg.norm(old_value, axis=-1)
        return jnp.mean(change / (old_norm + eps))

    def _encode_inputs(
        self,
        observations: jax.Array,
        goals: Optional[jax.Array],
    ) -> jax.Array:
        if goals is None:
            raise ValueError('HRM-Action-CoGHP requires a goal token.')

        observations = jnp.expand_dims(observations, axis=1)
        goals = jnp.expand_dims(goals, axis=1)
        if self.gc_encoder is not None:
            features = self.gc_encoder(
                observations,
                goals,
                goal_encoded=False,
                listwise=True,
            )
            observation_feature = self.feature_embed(features[0])
            goal_feature = features[1]
            return jnp.concatenate(
                [observation_feature, goal_feature],
                axis=1,
            )

        return jnp.concatenate(
            [self.feature_embed(observations), self.feature_embed(goals)],
            axis=1,
        )

    def _run_planner(
        self,
        features: jax.Array,
        teacher_subgoals: Optional[jax.Array],
        temperature: float,
        seed: jax.Array,
    ) -> tuple[list, jax.Array]:
        """Run the far-to-near planner one subgoal at a time.

        ``vanilla_coghp`` keeps the original Mixer path.  The matched and
        official-HRM variants invoke a fresh ``HRMCore`` carry for each
        subgoal, so the HRM recurrence refines that subgoal representation but
        does not replace CoGHP's outer autoregressive ordering.
        """

        batch_size = features.shape[0]
        query_tokens = jnp.tile(
            self.planner_query_tokens,
            (batch_size, 1, 1),
        )
        predicted_subgoals = jnp.zeros(
            (batch_size, self.num_subgoals, self.state_dim),
            dtype=features.dtype,
        )
        subgoal_keys = jax.random.split(seed, self.num_subgoals)
        high_dist_list = []

        for subgoal_idx in range(self.num_subgoals):
            if subgoal_idx == 0:
                previous_tokens = query_tokens
            else:
                history = (
                    predicted_subgoals[:, :subgoal_idx, :]
                    if teacher_subgoals is None
                    else teacher_subgoals[:, :subgoal_idx, :]
                )
                previous_tokens = jnp.concatenate(
                    [history, query_tokens[:, subgoal_idx:, :]],
                    axis=1,
                )

            planner_sequence = jnp.concatenate(
                [features, previous_tokens],
                axis=1,
            )
            if self.share_actor_core:
                _, planner_sequence = self.shared_actor_core(
                    None,
                    planner_sequence,
                )
            elif self.planner_core_type in ('vanilla_coghp', 'legacy_coghp'):
                for mixer_block in self.planner_mixer_blocks:
                    planner_sequence = mixer_block(planner_sequence)
            else:
                _, planner_sequence = self.planner_core(
                    None,
                    planner_sequence,
                )

            query_index = features.shape[1] + subgoal_idx
            high_dist = self.high_actor_head(
                planner_sequence[:, query_index, :],
                temperature=temperature,
            )
            high_dist_list.append(high_dist)
            predicted_subgoals = predicted_subgoals.at[
                :, subgoal_idx, :
            ].set(high_dist.sample(seed=subgoal_keys[subgoal_idx]))

        return high_dist_list, predicted_subgoals

    def _resolve_action_chain(
        self,
        features: jax.Array,
        subgoal_reps: Optional[jax.Array],
        predicted_subgoals: jax.Array,
        temperature: float,
        action_chain_source: Optional[str],
        seed: jax.Array,
    ) -> jax.Array:
        source = action_chain_source
        if source is None:
            source = (
                'predicted'
                if subgoal_reps is None
                else self.train_action_chain_source
            )
        if source not in ('teacher', 'predicted'):
            raise ValueError(
                'action_chain_source must be teacher or predicted; got '
                f'{source!r}.'
            )
        if source == 'teacher':
            if subgoal_reps is None:
                raise ValueError(
                    'Teacher action chain requested without subgoal_reps.'
                )
            return subgoal_reps
        if subgoal_reps is None:
            return predicted_subgoals

        # High-level loss remains teacher-forced, while this independent pass
        # produces the free-running chain used by action validation/ablation.
        _, free_running_subgoals = self._run_planner(
            features,
            teacher_subgoals=None,
            temperature=temperature,
            seed=seed,
        )
        return free_running_subgoals

    def __call__(
        self,
        observations: jax.Array,
        goals: jax.Array,
        seed: Optional[jax.Array] = None,
        subgoal_reps: Optional[jax.Array] = None,
        action_seq: Optional[jax.Array] = None,
        temperature: float = 1.0,
        action_chain_source: Optional[str] = None,
        return_diagnostics: bool = False,
    ):
        """Generate subgoals and one primitive-action distribution."""

        del action_seq
        if seed is None:
            seed = jax.random.PRNGKey(0)
        planner_seed, free_planner_seed, action_seed = jax.random.split(seed, 3)

        features = self._encode_inputs(observations, goals)
        if features.shape[1] != 2:
            raise ValueError(
                'Expected exactly observation and goal feature tokens; got '
                f'{features.shape[1]} tokens.'
            )
        high_dist_list, predicted_subgoals = self._run_planner(
            features,
            teacher_subgoals=subgoal_reps,
            temperature=temperature,
            seed=planner_seed,
        )
        action_chain = self._resolve_action_chain(
            features,
            subgoal_reps,
            predicted_subgoals,
            temperature,
            action_chain_source,
            free_planner_seed,
        )

        batch_size = features.shape[0]
        action_query = jnp.tile(
            self.action_query_token,
            (batch_size, 1, 1),
        )
        action_tokens = jnp.concatenate(
            [features, action_chain, action_query],
            axis=1,
        )
        expected_shape = (
            batch_size,
            self.num_subgoals + 3,
            self.state_dim,
        )
        if action_tokens.shape != expected_shape:
            raise ValueError(
                f'Expected action token shape {expected_shape}, got '
                f'{action_tokens.shape}.'
            )

        # Exactly one outer action-core call.  Vanilla CoGHP is a stateless
        # Mixer stack; the other variants use fresh transient HRM state per
        # action.  Only the HRM variant owns z_L/z_H constants.
        if self.share_actor_core:
            initial_carry = self.shared_actor_core.initial_carry((batch_size,))
            final_carry, core_output = self.shared_actor_core(
                initial_carry,
                action_tokens,
            )
        elif self.action_core_type == 'vanilla_coghp':
            initial_carry = {}
            final_carry = {}
            core_output = action_tokens
            for mixer_block in self.action_mixer_blocks:
                core_output = mixer_block(core_output)
        else:
            initial_carry = self.action_hrm.initial_carry((batch_size,))
            final_carry, core_output = self.action_hrm(
                initial_carry,
                action_tokens,
            )
        action_representation = core_output[:, -1, :].astype(jnp.float32)
        action_dist = self.low_actor_head(
            action_representation,
            temperature=temperature,
        )
        predicted_actions = action_dist.sample(seed=action_seed)

        self.sow('intermediates', 'action_token_sequence', action_tokens)
        self.sow('intermediates', 'action_core_output', core_output)
        if self.action_core_type == 'hrm':
            self.sow('intermediates', 'action_z_high', core_output)
            self.sow('intermediates', 'action_z_low', final_carry['z_L'])
        elif self.action_core_type == 'single_state_iterative':
            self.sow('intermediates', 'action_single_state', core_output)
        self.sow(
            'intermediates',
            'action_representation',
            action_representation,
        )

        if not return_diagnostics:
            return high_dist_list, action_dist, predicted_actions

        if self.action_core_type == 'vanilla_coghp':
            low_calls = 0
            high_calls = 0
            total_block_calls = self.num_mixer_blocks
            input_injections = 0
        elif self.action_core_type == 'matched_feedforward':
            low_calls = 1
            high_calls = 1
            total_block_calls = 2
            input_injections = 2
        else:
            low_calls = self.action_hrm_h_cycles * self.action_hrm_l_cycles
            high_calls = self.action_hrm_h_cycles
            total_block_calls = low_calls + high_calls
            input_injections = low_calls
        diagnostics = {
            'action_core/outer_calls': jnp.asarray(1, dtype=jnp.int32),
            'action_core/high_block_calls': jnp.asarray(
                high_calls,
                dtype=jnp.int32,
            ),
            'action_core/low_block_calls': jnp.asarray(
                low_calls,
                dtype=jnp.int32,
            ),
            'action_core/total_block_calls': jnp.asarray(
                total_block_calls,
                dtype=jnp.int32,
            ),
            'action_core/input_injections': jnp.asarray(
                input_injections,
                dtype=jnp.int32,
            ),
            'action_core/token_sequence_length': jnp.asarray(
                action_tokens.shape[1],
                dtype=jnp.int32,
            ),
            'action_core/input_token_norm': self._mean_norm(action_tokens),
            'action_core/output_token_norm': self._mean_norm(core_output),
            'action_core/action_representation_norm': self._mean_norm(
                action_representation
            ),
            'action_core/action_mode_norm': self._mean_norm(
                action_dist.mode()
            ),
        }
        if self.action_core_type == 'single_state_iterative':
            diagnostics.update(
                {
                    'action_core/state_initial_norm': self._mean_norm(
                        initial_carry['u']
                    ),
                    'action_core/state_final_norm': self._mean_norm(
                        core_output
                    ),
                }
            )
        elif self.action_core_type == 'hrm':
            # Keep historical metric keys for existing dashboards.
            diagnostics.update(
                {
                    'action_hrm/outer_cycles': jnp.asarray(
                        1,
                        dtype=jnp.int32,
                    ),
                    'action_hrm/high_level_calls': jnp.asarray(
                        high_calls,
                        dtype=jnp.int32,
                    ),
                    'action_hrm/low_level_calls': jnp.asarray(
                        low_calls,
                        dtype=jnp.int32,
                    ),
                    'action_hrm/token_sequence_length': jnp.asarray(
                        action_tokens.shape[1],
                        dtype=jnp.int32,
                    ),
                    'action_hrm/input_token_norm': self._mean_norm(
                        action_tokens
                    ),
                    'action_hrm/z_high_initial_norm': self._mean_norm(
                        initial_carry['z_H']
                    ),
                    'action_hrm/z_low_initial_norm': self._mean_norm(
                        initial_carry['z_L']
                    ),
                    'action_hrm/z_high_final_norm': self._mean_norm(
                        core_output
                    ),
                    'action_hrm/z_low_final_norm': self._mean_norm(
                        final_carry['z_L']
                    ),
                    'action_hrm/z_high_relative_change': (
                        self._mean_relative_change(
                            core_output,
                            initial_carry['z_H'],
                        )
                    ),
                    'action_hrm/z_low_relative_change': (
                        self._mean_relative_change(
                            final_carry['z_L'],
                            initial_carry['z_L'],
                        )
                    ),
                    'action_hrm/action_representation_norm': (
                        self._mean_norm(action_representation)
                    ),
                    'action_hrm/action_mode_norm': self._mean_norm(
                        action_dist.mode()
                    ),
                }
            )
        return high_dist_list, action_dist, predicted_actions, diagnostics


__all__ = [
    'HRMActionCoGHPPolicyNetwork',
    'resolve_action_block_type',
]
