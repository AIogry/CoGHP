"""Four-way Global Potential / Position-Latent CoGHP experiment.

The full_state + legacy_value path delegates to the original agent verbatim.
Position commands retain low-level gradients, but all planner command inputs
(including the final goal) and high-level targets are detached. The shared
observation embedding intentionally receives both actor losses.
"""

import copy

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.hrm_action_coghp import HRMActionCoGHPAgent, get_config as hrm_config
from utils.flax_utils import ModuleDict, TrainState
from utils.navigation_commands import GlobalPotential, PositionCommand, PositionDecoder


def stable_awr(advantage, beta, cap=100.0):
    return jax.lax.stop_gradient(jnp.exp(jnp.minimum(beta * advantage, jnp.log(cap))))


def global_ivl_loss(values, current_targets, next_targets, rewards, masks, valid,
                    discount, expectile):
    """Exact legacy IVL twin-head/sign convention, with masked OTA labels."""
    sign_target = rewards + discount * masks * jnp.min(next_targets, axis=0)
    advantage = jax.lax.stop_gradient(sign_target - jnp.mean(current_targets, axis=0))
    targets = jax.lax.stop_gradient(rewards[None] + discount * masks[None] * next_targets)
    weights = jnp.where(advantage >= 0, expectile, 1.0 - expectile)
    errors = weights[None] * (targets - values) ** 2
    return (errors * valid[None]).sum() / jnp.maximum(valid.sum(), 1.0)


class NavigationCoGHPAgent(HRMActionCoGHPAgent):
    @property
    def position_mode(self):
        return self.config['command_content'] == 'position'

    @property
    def global_mode(self):
        return self.config['high_weight_source'] == 'global_ota'

    def position_encode(self, observations, positions, params=None):
        return self.network.select('position_encoder')(observations, positions, params=params)

    def command_labels(self, batch, params=None):
        observations = jnp.repeat(batch['observations'][:, None, :], self.config['num_subgoals'], axis=1)
        if self.position_mode:
            return self.position_encode(observations, batch['high_actor_targets'][..., :2], params)
        return self.network.select('goal_rep')(
            jnp.concatenate([observations, batch['high_actor_targets']], axis=-1), params=params)

    def policy(self, observations, goals, rng, commands=None, params=None,
               temperature=1.0, chain_source=None, return_latents=False,
               return_diagnostics=False):
        kwargs = {}
        if self.position_mode:
            kwargs = dict(goal_rep_override=self.position_encode(observations, goals[..., :2], params),
                          detach_planner_commands=True)
        return self.network.select('actor_mixer')(
            observations, goals, rng, subgoal_reps=commands, params=params,
            temperature=temperature, action_chain_source=chain_source,
            return_latents=return_latents, return_diagnostics=return_diagnostics, **kwargs)

    def high_advantages(self, batch):
        if self.global_mode:
            value = self.network.select('global_value')
            goals = batch['high_actor_goals'][..., :2]
        else:
            value = self.network.select('value')
            goals = batch['high_actor_goals']
        v = jnp.mean(jnp.asarray(value(batch['observations'], goals)), axis=0)
        return jnp.stack([
            jnp.mean(jnp.asarray(value(batch['high_actor_targets'][:, j], goals)), axis=0) - v
            for j in range(self.config['num_subgoals'])], axis=1)

    def multi_high_actor_loss(self, batch, dist_list, obs_expand, grad_params):
        if not self.position_mode and not self.global_mode:
            return super().multi_high_actor_loss(batch, dist_list, obs_expand, grad_params)
        targets = jax.lax.stop_gradient(self.command_labels(batch))
        advantages = self.high_advantages(batch)
        weights = stable_awr(advantages, self.config['high_alpha'], self.config['awr_max_weight'])
        loss, mse, logp, std = 0.0, 0.0, 0.0, 0.0
        info = {}
        k = self.config['num_subgoals']
        for j in range(k):
            lp = dist_list[j].log_prob(targets[:, j])
            loss += -(weights[:, j] * lp).mean() / self.config['subgoal_steps'] * self.config['high_discount'] ** (k - j - 1)
            mse += jnp.mean((dist_list[j].mode() - targets[:, j]) ** 2)
            logp += lp.mean()
            std += dist_list[j].scale_diag.mean()
            info[f'adv_{j}/mean'] = advantages[:, j].mean()
            info[f'adv_{j}/std'] = advantages[:, j].std()
            info[f'weight_{j}/cap_fraction'] = jnp.mean(weights[:, j] >= self.config['awr_max_weight'] * (1 - 1e-6))
            info[f'weight_{j}/mean'] = weights[:, j].mean()
            info[f'weight_{j}/ess_fraction'] = weights[:, j].sum() ** 2 / (weights.shape[0] * jnp.maximum(jnp.sum(weights[:, j] ** 2), 1e-12))
        info.update(actor_loss=loss, adv=advantages.mean(), mse=mse / k, bc_log_prob=logp / k, std=std / k)
        return loss / k, info

    def actor_loss(self, batch, grad_params, rng):
        if not self.position_mode:
            return super().actor_loss(batch, grad_params, rng)
        commands = self.command_labels(batch, grad_params)  # LIVE for Executor.
        high, low, _, diagnostics = self.policy(
            batch['observations'], batch['high_actor_goals'], rng, commands,
            grad_params, return_diagnostics=True)
        high_loss, high_info = self.multi_high_actor_loss(batch, high, None, grad_params)
        low_loss, low_info = self.low_actor_loss(batch, low)
        return high_loss + low_loss, high_info, low_info, diagnostics

    def position_loss(self, batch, params):
        positions = jnp.concatenate([batch['high_actor_goals'][:, None, :2],
                                     batch['high_actor_targets'][..., :2]], axis=1)
        states = jnp.broadcast_to(batch['observations'][:, None, :],
                                  (*positions.shape[:-1], batch['observations'].shape[-1]))
        latent = self.position_encode(states, positions, params)
        decoded = self.network.select('position_decoder')(latent, params=params)
        relative = (positions - states[..., :2]) / jnp.asarray(self.config['position_scale'])
        loss = jnp.mean((decoded - relative) ** 2)
        return loss, dict(reconstruction_mse=loss,
                          latent_norm=jnp.linalg.norm(latent, axis=-1).mean(),
                          conditional_spread=jnp.mean(jnp.var(latent, axis=1)))

    def global_loss(self, batch, params):
        goals = batch['global_goals']
        current_targets = self.network.select('target_global_value')(batch['observations'], goals)
        next_targets = self.network.select('target_global_value')(batch['global_observations'], goals)
        values = self.network.select('global_value')(batch['observations'], goals, params=params)
        loss = global_ivl_loss(values, current_targets, next_targets,
                               batch['global_rewards'], batch['global_masks'], batch['global_valid'],
                               self.config['global_discount'], self.config['expectile'])
        valid = batch['global_valid']
        den = jnp.maximum(valid.sum(), 1)
        return loss, dict(loss=loss, v_mean=values.mean(), v_min=values.min(), v_max=values.max(),
                          excluded_fraction=1 - valid.mean(),
                          success_fraction=((1 - batch['global_masks']) * valid).sum() / den,
                          mean_duration=(batch['global_offsets'] * valid).sum() / den)

    def loss_components(self, batch, params, rng):
        value, vi = self.value_loss(batch, params)
        actor, hi, li, di = self.actor_loss(batch, params, rng)
        # The high logging entry historically lacks division by K.
        high = hi['actor_loss'] / self.config['num_subgoals']
        low = li['actor_loss']
        info = {**{f'value/{k}': v for k, v in vi.items()},
                **{f'high_actor/{k}': v for k, v in hi.items()},
                **{f'low_actor/{k}': v for k, v in li.items()}, **di}
        pos, glob = jnp.asarray(0.0), jnp.asarray(0.0)
        if self.position_mode:
            pos, pi = self.position_loss(batch, params)
            info.update({f'position/{k}': v for k, v in pi.items()})
        if self.global_mode:
            glob, gi = self.global_loss(batch, params)
            info.update({f'global/{k}': v for k, v in gi.items()})
        return dict(value=value, high=high, low=low, position=pos, global_value=glob, actor=actor), info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        if not self.position_mode and not self.global_mode:
            return super().total_loss(batch, grad_params, rng)
        losses, info = self.loss_components(batch, grad_params, self.rng if rng is None else rng)
        return (losses['value'] + losses['actor'] + losses['global_value']
                + self.config['position_loss_coef'] * losses['position']), info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        network, info = self.network.apply_loss_fn(lambda p: self.total_loss(batch, p, rng))
        # Match legacy's pre-optimizer online parameters in its EMA exactly.
        self.target_update(network, 'value')
        if self.global_mode:
            self.target_update(network, 'global_value')
        return self.replace(network=network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        if not self.position_mode:
            return super().sample_actions(observations, goals, seed, temperature)
        _, _, actions = self.policy(observations[None], goals[None], seed, temperature=temperature)
        return jnp.clip(actions[0], -1, 1)

    @jax.jit
    def sample_with_commands(self, observations, goals, positions, seed):
        commands = self.position_encode(jnp.broadcast_to(observations, (len(positions), len(observations))), positions)
        _, dist, _ = self.policy(observations[None], goals[None], seed, commands[None],
                                 temperature=0.0, chain_source='teacher')
        return jnp.clip(dist.mode()[0], -1, 1)

    @jax.jit
    def predicted_positions(self, observations, goals, seed):
        _, _, _, predicted = self.policy(observations[None], goals[None], seed,
                                          temperature=0.0, return_latents=True)
        offsets = self.network.select('position_decoder')(predicted[0])
        return observations[:2] + jnp.asarray(self.config['position_scale']) * offsets

    @jax.jit
    def position_probe(self, batch):
        return self.command_labels(batch)

    @jax.jit
    def validation_rollout_info(self, batch, rng=None):
        if not self.position_mode:
            return super().validation_rollout_info(batch, rng)
        rng = self.rng if rng is None else rng
        commands = self.command_labels(batch)
        _, teacher, _ = self.policy(batch['observations'], batch['high_actor_goals'], rng,
                                     commands, temperature=0.0, chain_source='teacher')
        _, predicted, _, latent = self.policy(batch['observations'], batch['high_actor_goals'], rng,
                                               temperature=0.0, return_latents=True)
        tmse = jnp.mean((teacher.mode() - batch['actions']) ** 2)
        pmse = jnp.mean((predicted.mode() - batch['actions']) ** 2)
        scale = jnp.asarray(self.config['position_scale'])
        states = jnp.broadcast_to(batch['observations'][:, None], (*commands.shape[:-1], batch['observations'].shape[-1]))
        target = batch['high_actor_targets'][..., :2]
        out = {'validation_action/teacher/mse': tmse, 'validation_action/predicted/mse': pmse,
               'validation_action/gap/mse': pmse - tmse,
               'validation_position/predicted/latent_mse': jnp.mean((latent - commands) ** 2)}
        for name, z in [('teacher', commands), ('predicted', latent)]:
            q = states[..., :2] + scale * self.network.select('position_decoder')(z)
            recoded = self.position_encode(states, q)
            out[f'validation_position/{name}/position_error'] = jnp.linalg.norm(q - target, axis=-1).mean()
            out[f'validation_position/{name}/cycle_mse'] = jnp.mean((recoded - z) ** 2)
            out[f'validation_position/{name}/norm'] = jnp.linalg.norm(z, axis=-1).mean()
        return out

    @jax.jit
    def gradient_diagnostics(self, batch):
        out = {}
        if self.position_mode:
            for loss_name in ('high', 'low', 'position'):
                grads = jax.grad(lambda p: self.loss_components(batch, p, self.rng)[0][loss_name])(self.network.params)
                for group in ('position_encoder', 'position_decoder'):
                    leaves = jax.tree_util.tree_leaves(grads[f'modules_{group}'])
                    out[f'gradient/{loss_name}/{group}'] = jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves))
        return out

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        if config['command_content'] not in ('full_state', 'position') or config['high_weight_source'] not in ('legacy_value', 'global_ota'):
            raise ValueError('Unknown navigation condition.')
        if config['share_actor_core'] or config['train_action_chain_source'] != 'teacher':
            raise ValueError('This experiment requires split cores and teacher-forced action training.')
        if config['encoder'] is not None or config['frame_stack'] is not None:
            raise ValueError('This study supports unstacked state-based navigation only.')
        base = HRMActionCoGHPAgent.create(seed, ex_observations, ex_actions, config)
        position = config['command_content'] == 'position'
        glob = config['high_weight_source'] == 'global_ota'
        if not position and not glob:
            return cls(base.rng, base.network, base.config)
        scale = tuple(config['position_scale'])
        mean = tuple(config['position_mean'])
        if len(scale) != 2 or not np.isfinite(scale).all() or min(scale) <= 0:
            raise ValueError('Finite positive training position scales required.')
        modules = dict(base.network.model_def.modules)
        args = dict(goal_rep=jnp.concatenate([ex_observations, ex_observations], axis=-1),
                    value=(ex_observations, ex_observations),
                    target_value=(ex_observations, ex_observations),
                    actor_mixer=(ex_observations, ex_observations, base.rng))
        if position:
            modules['position_encoder'] = PositionCommand(config['feature_dim'], scale, tuple(config['position_hidden_dims']))
            modules['position_decoder'] = PositionDecoder(tuple(config['position_hidden_dims']))
            args['position_encoder'] = (ex_observations, ex_observations[..., :2])
            args['position_decoder'] = jnp.ones((*ex_observations.shape[:-1], config['feature_dim']))
        if glob:
            for name in ('global_value', 'target_global_value'):
                modules[name] = GlobalPotential(config['global_dim'], mean, scale, tuple(config['global_hidden_dims']))
                args[name] = (ex_observations, ex_observations[..., :2])
        definition = ModuleDict(modules)
        variables = definition.init(jax.random.fold_in(jax.random.PRNGKey(seed), 9127), **args)
        params = variables['params']
        # Existing parameters/constants start identically across all conditions.
        params.update(copy.deepcopy(base.network.params))
        if glob:
            params['modules_target_global_value'] = copy.deepcopy(params['modules_global_value'])
        state = {k: v for k, v in variables.items() if k != 'params'}
        for collection, value in flax.core.unfreeze(base.network.model_state).items():
            state.setdefault(collection, {}).update(value)
        network = TrainState.create(definition, params, tx=optax.adam(config['lr']), model_state=state)
        return cls(base.rng, network, flax.core.FrozenDict(config))


def get_config():
    config = hrm_config()
    config.agent_name = 'navigation_coghp'
    config.command_content = 'full_state'
    config.high_weight_source = 'legacy_value'
    config.position_scale = (1.0, 1.0)  # Replaced from TRAIN data before agent creation.
    config.position_mean = (0.0, 0.0)
    config.position_hidden_dims = (128, 128)
    config.position_loss_coef = 1.0
    config.global_dim = 128
    config.global_hidden_dims = (512, 512, 512)
    config.global_abstraction_steps = 25
    config.global_discount = 0.99
    config.global_goal_tolerance = -1.0  # Resolved from the actual environment.
    config.awr_max_weight = 100.0
    config.navigation_protocol = 'position_ota_v1'
    return config
