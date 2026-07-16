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

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
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
    return config
