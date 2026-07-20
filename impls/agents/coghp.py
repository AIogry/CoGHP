import copy
from typing import Any

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize
from utils.coghp_network import HierarchicalPolicyNetwork


class CoGHPAgent(flax.struct.PyTreeNode):
    """CoGHP agent"""
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    all_time_actions: Any = None #list = flax.struct.field(default_factory=list)
    goal_temp: Any = None

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']  # shape: (B, D)
        actions = batch['actions']  # ground truth or teacher-forcing tokens
        goals = batch['high_actor_goals']

        obs_expand = jnp.expand_dims(observations, axis=1)  # (B, 1, D)
        obs_expand = jnp.repeat(obs_expand, self.config["num_subgoals"], axis=1)
        subgoals_reps = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1),
            params=grad_params,
        )

        high_dist, low_dist, _ = self.network.select('actor_mixer')(observations,
                                                                    goals,
                                                                    rng,
                                                                    subgoal_reps=subgoals_reps,
                                                                    action_seq=actions,
                                                                    params=grad_params)

        if self.config['num_subgoals'] > 0:
            high_actor_loss, high_actor_info = self.multi_high_actor_loss(batch, high_dist, obs_expand, grad_params)
        else:
            high_actor_loss = 0.0
            high_actor_info = {}
        
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, low_dist)

        actor_loss = high_actor_loss + low_actor_loss
                
        return actor_loss, high_actor_info, low_actor_info
    
    def multi_high_actor_loss(self, batch, dist_list, obs_expand, grad_params):
        multi_targets = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1),
        )

        actor_loss, adv_mean, bc_log_prob, mse, std = 0, 0, 0, 0, 0

        for i in range(self.config['num_subgoals']):
            v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
            nv1, nv2 = self.network.select('value')(batch['high_actor_targets'][:, i, :], batch['high_actor_goals'])
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v

            exp_a = jnp.exp(adv * self.config['high_alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            target = multi_targets[:, i, :]

            log_prob = dist_list[i].log_prob(target)

            actor_loss += -((exp_a * log_prob).mean() / self.config['subgoal_steps']) * (self.config['high_discount'] ** (self.config['num_subgoals'] - i - 1))

            adv_mean += adv.mean()
            bc_log_prob += log_prob.mean()
            mse += jnp.mean((dist_list[i].mode() - target) ** 2)
            std += jnp.mean(dist_list[i].scale_diag)

        return actor_loss / self.config['num_subgoals'], {
            'actor_loss': actor_loss,
            'adv': adv_mean / self.config['num_subgoals'],
            'bc_log_prob': bc_log_prob / self.config['num_subgoals'],
            'mse': mse / self.config['num_subgoals'],
            'std': std / self.config['num_subgoals'],
        }
    
    def low_actor_loss(self, batch, dist):
        if self.config['num_subgoals'] > 0:
            target = batch['high_actor_targets'][:, -1, :]
        else:
            target = batch['high_actor_goals']
            
        v1, v2 = self.network.select('value')(batch['observations'], target)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], target)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        action = batch['actions']

        log_prob = dist.log_prob(action)
        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }

        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - action) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        actor_loss, high_actor_info, low_actor_info = self.actor_loss(batch, grad_params, rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = value_loss + actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor network."""
        high_seed, low_seed = jax.random.split(seed)

        observations = jnp.expand_dims(observations, axis=0)  # (1, D)
        
        if goals is not None:
            goals = jnp.expand_dims(goals, axis=0)
        
        high_dist, _, predicted_actions = self.network.select('actor_mixer')(
            observations, goals, seed, subgoal_reps=None, action_seq=None, temperature=temperature
        )

        goal_info = {}

        actions = predicted_actions[0]

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new CoGHP agent.
        
        Args:
            seed: Random seed.
            ex_observations: Example observation batch.
            ex_actions: Example action batch (for discrete actions, expect max action value).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        ex_latent_goals = jnp.zeros((ex_observations.shape[0], config['feature_dim']))
        if config['discrete']:
            action_dim = jnp.max(ex_actions) + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        # Define encoder.
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
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())

        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None
        
        # Define value and actor networks.
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
        else:
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
        
        if config['discrete']:
            actor_mixer_def = None
            raise NotImplementedError("Discrete actions not supported yet.")
        
        else:
            if config['gc_enc'] == 'concat':
                gc_enc = low_actor_encoder_def
            else:
                gc_enc = high_actor_encoder_def

            actor_mixer_def = HierarchicalPolicyNetwork(
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
                causal_mixer=config.get('causal_mixer', True),
                action_use_full_subgoal_chain=config.get('action_use_full_subgoal_chain', True),
                share_mixer_weights=config.get('share_mixer_weights', False),
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

        print("Creating Done")

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='coghp',  # Agent name.
            lr=3e-4,              # Learning rate.
            batch_size=256,      # Batch size.
            discount=0.99,
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            alpha=1.0,
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            high_const_std=True,
            low_const_std=True,
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            dataset_class='MultiHGCDataset',
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),

            feature_dim=32,
            mixer_hidden=32,
            num_mixer_blocks=1,
            enc_hidden_dims=(512, 512, 512),
            layer_norm=True,  # Whether to use layer normalization.
            action_chunk=ml_collections.config_dict.placeholder(int),
            num_subgoals=1,
            causal_mixer=True,
            action_use_full_subgoal_chain=True,
            share_mixer_weights=False,
            gc_enc='concat',
            high_discount=0.8,
        )
    )
    return config
