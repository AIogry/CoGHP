"""CoGHP with shared-block feed-forward or recurrent action cores."""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.coghp import CoGHPAgent, get_config as get_coghp_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.hrm_action_coghp_network import HRMActionCoGHPPolicyNetwork
from utils.networks import GCActor, GCValue, Identity, LengthNormalize, MLP


class HRMActionCoGHPAgent(CoGHPAgent):
    """CoGHP planner with a configurable primitive-action core."""

    def actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        goals = batch['high_actor_goals']
        observation_sequence = jnp.repeat(
            observations[:, None, :],
            self.config['num_subgoals'],
            axis=1,
        )
        teacher_subgoals = self.network.select('goal_rep')(
            jnp.concatenate(
                [observation_sequence, batch['high_actor_targets']],
                axis=-1,
            ),
            params=grad_params,
        )

        high_dists, action_dist, _, diagnostics = self.network.select(
            'actor_mixer'
        )(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            action_seq=batch['actions'],
            params=grad_params,
            return_diagnostics=True,
        )
        high_actor_loss, high_actor_info = self.multi_high_actor_loss(
            batch,
            high_dists,
            observation_sequence,
            grad_params,
        )
        low_actor_loss, low_actor_info = self.low_actor_loss(
            batch,
            action_dist,
        )
        return (
            high_actor_loss + low_actor_loss,
            high_actor_info,
            low_actor_info,
            diagnostics,
        )

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for key, value in value_info.items():
            info[f'value/{key}'] = value

        actor_loss, high_info, low_info, diagnostics = self.actor_loss(
            batch,
            grad_params,
            rng,
        )
        for key, value in high_info.items():
            info[f'high_actor/{key}'] = value
        for key, value in low_info.items():
            info[f'low_actor/{key}'] = value
        info.update(diagnostics)
        return value_loss + actor_loss, info

    def validation_rollout_info(self, batch, rng=None):
        """Measure the action error under teacher and free-running chains."""

        rng = self.rng if rng is None else rng
        observations = batch['observations']
        goals = batch['high_actor_goals']
        observation_sequence = jnp.repeat(
            observations[:, None, :],
            self.config['num_subgoals'],
            axis=1,
        )
        teacher_subgoals = self.network.select('goal_rep')(
            jnp.concatenate(
                [observation_sequence, batch['high_actor_targets']],
                axis=-1,
            )
        )

        _, teacher_dist, _ = self.network.select('actor_mixer')(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=0.0,
            action_chain_source='teacher',
        )
        _, predicted_dist, _ = self.network.select('actor_mixer')(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=0.0,
            action_chain_source='predicted',
        )
        _, teacher_density_dist, _ = self.network.select('actor_mixer')(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=1.0,
            action_chain_source='teacher',
        )
        _, predicted_density_dist, _ = self.network.select('actor_mixer')(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=1.0,
            action_chain_source='predicted',
        )

        actions = batch['actions']
        teacher_mse = jnp.mean((teacher_dist.mode() - actions) ** 2)
        predicted_mse = jnp.mean((predicted_dist.mode() - actions) ** 2)
        return {
            'validation_action/teacher/mse': teacher_mse,
            'validation_action/predicted/mse': predicted_mse,
            'validation_action/gap/mse': predicted_mse - teacher_mse,
            'validation_action/teacher/bc_log_prob': (
                teacher_density_dist.log_prob(actions).mean()
            ),
            'validation_action/predicted/bc_log_prob': (
                predicted_density_dist.log_prob(actions).mean()
            ),
        }

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        observations = observations[None, :]
        if goals is not None:
            goals = goals[None, :]
        _, _, predicted_actions = self.network.select('actor_mixer')(
            observations,
            goals,
            seed,
            subgoal_reps=None,
            action_seq=None,
            temperature=temperature,
        )
        actions = predicted_actions[0]
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config['num_subgoals'] < 1:
            raise ValueError('HRM-Action-CoGHP requires num_subgoals >= 1.')
        if config['train_action_chain_source'] not in ('teacher', 'predicted'):
            raise ValueError(
                'train_action_chain_source must be teacher or predicted.'
            )
        if config['action_hrm_forward_dtype'] not in (
            'float32',
            'bfloat16',
        ):
            raise ValueError(
                'action_hrm_forward_dtype must be float32 or bfloat16.'
            )
        if config['discrete']:
            raise NotImplementedError('Discrete actions are not supported.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_modules = [encoder_module()]
        else:
            encoder_module = None
            goal_rep_modules = []
        goal_rep_modules.extend(
            [
                MLP(
                    hidden_dims=(
                        *config['enc_hidden_dims'],
                        config['feature_dim'],
                    ),
                    activate_final=False,
                    layer_norm=config['layer_norm'],
                ),
                LengthNormalize(),
            ]
        )
        goal_rep_def = nn.Sequential(goal_rep_modules)

        if encoder_module is not None:
            value_encoder_def = GCEncoder(
                state_encoder=encoder_module(),
                concat_encoder=goal_rep_def,
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=encoder_module(),
                concat_encoder=goal_rep_def,
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=encoder_module(),
                concat_encoder=goal_rep_def,
            )
            high_actor_encoder_def = GCEncoder(
                concat_encoder=encoder_module()
            )
        else:
            value_encoder_def = GCEncoder(
                state_encoder=Identity(),
                concat_encoder=goal_rep_def,
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=Identity(),
                concat_encoder=goal_rep_def,
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=Identity(),
                concat_encoder=goal_rep_def,
            )
            high_actor_encoder_def = None

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['low_const_std'],
            gc_encoder=None,
        )
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['feature_dim'],
            state_dependent_std=False,
            const_std=config['high_const_std'],
            gc_encoder=None,
        )
        gc_encoder = (
            low_actor_encoder_def
            if config['gc_enc'] == 'concat'
            else high_actor_encoder_def
        )

        actor_mixer_def = HRMActionCoGHPPolicyNetwork(
            state_dim=config['feature_dim'],
            high_actor_head=high_actor_def,
            low_actor_head=low_actor_def,
            num_mixer_blocks=config['num_mixer_blocks'],
            mixer_token_hidden=config['mixer_hidden'],
            mixer_channel_hidden=config['mixer_hidden'],
            gc_encoder=gc_encoder,
            enc_hidden=config['enc_hidden_dims'],
            num_subgoals=config['num_subgoals'],
            causal_mixer=config.get('causal_mixer', True),
            action_hrm_num_layers=config['action_hrm_num_layers'],
            action_hrm_intermediate_size=(
                config['action_hrm_intermediate_size']
            ),
            action_hrm_head_dim=config['action_hrm_head_dim'],
            action_hrm_h_cycles=config['action_hrm_h_cycles'],
            action_hrm_l_cycles=config['action_hrm_l_cycles'],
            action_hrm_bptt=config['action_hrm_bptt'],
            action_hrm_is_causal=config['action_hrm_is_causal'],
            action_hrm_is_mlp_mixer=(
                config['action_hrm_is_mlp_mixer']
            ),
            action_hrm_mlp_mixer_intermediate_size=(
                config['action_hrm_mlp_mixer_intermediate_size']
            ),
            action_core_type=config.get('action_core_type', 'hrm'),
            action_block_type=config.get('action_block_type', 'auto'),
            hrm_block_type=config.get('hrm_block_type', 'legacy'),
            hrm_token_hidden=config.get('hrm_token_hidden', 32),
            hrm_channel_hidden=config.get('hrm_channel_hidden', 32),
            hrm_mixer_intermediate_size=config.get(
                'hrm_mixer_intermediate_size',
                -1,
            ),
            hrm_layernorm_affine=config.get(
                'hrm_layernorm_affine',
                True,
            ),
            action_hrm_norm_eps=config['action_hrm_norm_eps'],
            action_hrm_rope_theta=config['action_hrm_rope_theta'],
            action_hrm_forward_dtype=config['action_hrm_forward_dtype'],
            train_action_chain_source=config['train_action_chain_source'],
        )

        network_info = {
            'goal_rep': (
                goal_rep_def,
                jnp.concatenate([ex_observations, ex_goals], axis=-1),
            ),
            'value': (value_def, (ex_observations, ex_goals)),
            'target_value': (
                target_value_def,
                (ex_observations, ex_goals),
            ),
            'actor_mixer': (
                actor_mixer_def,
                (ex_observations, ex_goals, rng),
            ),
        }
        network_def = ModuleDict(
            {key: value[0] for key, value in network_info.items()}
        )
        network_args = {
            key: value[1] for key, value in network_info.items()
        }
        network_variables = network_def.init(init_rng, **network_args)
        network_params = network_variables['params']
        expected_action_core_params = config.get(
            'expected_action_core_params',
            -1,
        )
        if expected_action_core_params >= 0:
            action_core_params = network_params['modules_actor_mixer'][
                'action_hrm'
            ]
            actual_action_core_params = sum(
                int(leaf.size)
                for leaf in jax.tree_util.tree_leaves(action_core_params)
            )
            if actual_action_core_params != expected_action_core_params:
                raise ValueError(
                    'Action-core parameter contract failed: expected '
                    f'{expected_action_core_params}, got '
                    f'{actual_action_core_params}.'
                )
        network_params['modules_target_value'] = network_params[
            'modules_value'
        ]
        model_state = {
            collection: variables
            for collection, variables in network_variables.items()
            if collection != 'params'
        }
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
            model_state=model_state,
        )
        print('Creating official HRM-Action-CoGHP Done')
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = get_coghp_config()
    config.agent_name = 'hrm_action_coghp'

    config.action_hrm_num_layers = 1
    config.action_hrm_intermediate_size = 512
    config.action_hrm_head_dim = 32
    config.action_hrm_h_cycles = 2
    config.action_hrm_l_cycles = 2
    config.action_hrm_bptt = True
    config.action_hrm_is_causal = True
    config.action_hrm_is_mlp_mixer = True
    config.action_hrm_mlp_mixer_intermediate_size = 256
    # These canonical selectors separate computation topology from the shared
    # H/L block family.  ``auto`` lets existing hrm_block_type and the oldest
    # action_hrm_is_mlp_mixer configs retain their original meaning.
    config.action_core_type = 'hrm'
    config.action_block_type = 'auto'
    # Keep the legacy block selector neutral so an old command that changes
    # only action_hrm_is_mlp_mixer still selects the same mixer/attention
    # parameter tree.  A mixer size of -1 inherits the old
    # action_hrm_mlp_mixer_intermediate_size setting.
    config.hrm_block_type = 'legacy'
    config.hrm_token_hidden = 32
    config.hrm_channel_hidden = 32
    config.hrm_mixer_intermediate_size = -1
    config.hrm_layernorm_affine = True
    config.expected_action_core_params = -1
    config.action_hrm_norm_eps = 1e-6
    config.action_hrm_rope_theta = 10000.0
    config.action_hrm_forward_dtype = 'float32'

    config.train_action_chain_source = 'teacher'
    config.enable_action_hrm_validation = True
    return config


__all__ = ['HRMActionCoGHPAgent', 'get_config']
