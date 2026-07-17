import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.coghp import CoGHPAgent, get_config as get_coghp_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.hcoghp_network import HCoGHPPolicyNetwork
from utils.networks import GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, MLP


class HCoGHPAgent(CoGHPAgent):
    """CoGHP agent with a shared-weight HRM-style actor mixer."""

    def get_scheduled_sampling_prob(self, step):
        """Return the linear scheduled sampling probability for the current training step."""
        step = jnp.asarray(step, dtype=jnp.float32)
        start = jnp.asarray(self.config.get('scheduled_sampling_start_step', 100000), dtype=jnp.float32)
        end = jnp.asarray(self.config.get('scheduled_sampling_end_step', 300000), dtype=jnp.float32)
        max_prob = jnp.asarray(self.config.get('scheduled_sampling_max_prob', 0.5), dtype=jnp.float32)
        enabled = jnp.asarray(self.config.get('scheduled_sampling_enabled', False), dtype=jnp.float32)
        progress = (step - start) / jnp.maximum(end - start, 1.0)
        progress = jnp.clip(progress, 0.0, 1.0)
        return enabled * max_prob * progress

    def actor_loss(self, batch, grad_params, rng, scheduled_sampling_prob=0.0):
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['high_actor_goals']

        obs_expand = jnp.expand_dims(observations, axis=1)
        obs_expand = jnp.repeat(obs_expand, self.config['num_subgoals'], axis=1)
        subgoals_reps = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1),
            params=grad_params,
        )

        high_dist, low_dist, _, sampling_info = self.network.select('actor_mixer')(
            observations,
            goals,
            rng,
            subgoal_reps=subgoals_reps,
            action_seq=actions,
            params=grad_params,
            scheduled_sampling_prob=scheduled_sampling_prob,
            scheduled_sampling_use_mode=self.config.get('scheduled_sampling_use_mode', True),
            scheduled_sampling_stop_gradient=self.config.get('scheduled_sampling_stop_gradient', True),
            return_sampling_diagnostics=True,
        )

        if self.config['num_subgoals'] > 0:
            high_actor_loss, high_actor_info = self.multi_high_actor_loss(batch, high_dist, obs_expand, grad_params)
        else:
            high_actor_loss = 0.0
            high_actor_info = {}

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, low_dist)
        actor_loss = high_actor_loss + low_actor_loss

        return actor_loss, high_actor_info, low_actor_info, sampling_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        scheduled_sampling_prob = self.get_scheduled_sampling_prob(self.network.step)
        actor_loss, high_actor_info, low_actor_info, sampling_info = self.actor_loss(
            batch,
            grad_params,
            rng,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v
        for k, v in sampling_info.items():
            if k == 'scheduled_sampling_prob':
                info['scheduled_sampling/prob'] = v
            elif k == 'scheduled_sampling_ratio':
                info['scheduled_sampling/actual_ratio'] = v
            elif k.startswith('scheduled_sampling_ratio_'):
                token_idx = k.rsplit('_', 1)[-1]
                info[f'scheduled_sampling/subgoal_{token_idx}_ratio'] = v

        loss = value_loss + actor_loss
        return loss, info

    def validation_rollout_info(self, batch, rng=None):
        """Compute teacher-forced and free-running validation diagnostics.

        This method is read-only: it does not participate in the training loss
        and does not change the actor forward contract used by updates.
        """
        rng = rng if rng is not None else self.rng
        observations = batch['observations']
        goals = batch['high_actor_goals']
        obs_expand = jnp.expand_dims(observations, axis=1)
        obs_expand = jnp.repeat(obs_expand, self.config['num_subgoals'], axis=1)
        target_reps = self.network.select('goal_rep')(jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1))

        def collect(prefix, subgoal_reps, return_diagnostics=False):
            result = self.network.select('actor_mixer')(
                observations,
                goals,
                rng,
                subgoal_reps=subgoal_reps,
                action_seq=None,
                return_diagnostics=return_diagnostics,
                scheduled_sampling_prob=0.0,
                return_sampling_diagnostics=False,
            )
            diagnostics = None
            if return_diagnostics:
                high_dist_list, low_dist, _, diagnostics = result
            else:
                high_dist_list, low_dist, _ = result

            info = {}
            subgoal_mses = []
            for i, high_dist in enumerate(high_dist_list):
                subgoal_mse = jnp.mean((high_dist.mode() - target_reps[:, i, :]) ** 2)
                info[f'{prefix}/subgoal_{i}_mse'] = subgoal_mse
                subgoal_mses.append(subgoal_mse)

            if subgoal_mses:
                info[f'{prefix}/high_actor/mse'] = jnp.mean(jnp.stack(subgoal_mses))

            action_mse = jnp.mean((low_dist.mode() - batch['actions']) ** 2)
            info[f'{prefix}/action_mse'] = action_mse
            info[f'{prefix}/low_actor/mse'] = action_mse

            if diagnostics is not None:
                for key, value in diagnostics.items():
                    info[f'diagnostics/{key}'] = value

            return info

        info = collect(
            'validation_teacher',
            subgoal_reps=target_reps,
            return_diagnostics=self.config.get('enable_hcoghp_diagnostics', False),
        )

        if self.config.get('enable_free_running_validation', False):
            free_info = collect('validation_free', subgoal_reps=None)
            info.update(free_info)
            if self.config['num_subgoals'] > 0:
                info['validation_gap/high_actor_mse'] = (
                    free_info['validation_free/high_actor/mse'] - info['validation_teacher/high_actor/mse']
                )
                for i in range(self.config['num_subgoals']):
                    info[f'validation_gap/subgoal_{i}_mse'] = (
                        free_info[f'validation_free/subgoal_{i}_mse'] - info[f'validation_teacher/subgoal_{i}_mse']
                    )
            info['validation_gap/low_actor_mse'] = (
                free_info['validation_free/low_actor/mse'] - info['validation_teacher/low_actor/mse']
            )
            info['validation_gap/action_mse'] = free_info['validation_free/action_mse'] - info['validation_teacher/action_mse']

        return info

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        if config.get('scheduled_sampling_schedule', 'linear') != 'linear':
            raise ValueError('HCoGHP scheduled sampling currently supports only the linear schedule.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = jnp.max(ex_actions) + 1
        else:
            action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []

        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['enc_hidden_dims'], config['feature_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
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

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=1,
                gc_encoder=None,
            )
            raise NotImplementedError('Discrete actions not supported yet.')

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

        if config['gc_enc'] == 'concat':
            gc_enc = low_actor_encoder_def
        else:
            gc_enc = high_actor_encoder_def

        actor_mixer_def = HCoGHPPolicyNetwork(
            num_tokens=1,
            state_dim=config['feature_dim'],
            num_action_dims=action_dim,
            joint_embed_dim=config['feature_dim'],
            num_mixer_blocks=config['num_mixer_blocks'],
            mixer_token_hidden=config['mixer_hidden'],
            mixer_channel_hidden=config['mixer_hidden'],
            gc_encoder=gc_enc,
            layer_norm=config['layer_norm'],
            high_actor_head=high_actor_def,
            low_actor_head=low_actor_def,
            enc_hidden=config['enc_hidden_dims'],
            num_subgoals=config['num_subgoals'],
            hrm_l_cycles=config['hrm_l_cycles'],
            hrm_mix_scale=config['hrm_mix_scale'],
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            actor_mixer=(actor_mixer_def, (ex_observations, ex_goals, rng)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        print('Creating HCoGHP Done')

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_coghp_config()
    config.agent_name = 'hcoghp'
    config.hrm_l_cycles = 2
    config.hrm_mix_scale = True
    config.enable_hcoghp_diagnostics = False
    config.enable_free_running_validation = False
    config.scheduled_sampling_enabled = False
    config.scheduled_sampling_start_step = 100000
    config.scheduled_sampling_end_step = 300000
    config.scheduled_sampling_max_prob = 0.5
    config.scheduled_sampling_use_mode = True
    config.scheduled_sampling_stop_gradient = True
    config.scheduled_sampling_schedule = 'linear'
    return config
