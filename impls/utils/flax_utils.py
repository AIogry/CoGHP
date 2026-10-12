import functools
import glob
import json
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        model_state: Immutable non-parameter Flax variable collections, such
            as ``constants``.  These variables are passed to ``Module.apply``
            but are not differentiated or optimized.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    model_state: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, model_state=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None
        if model_state is None:
            model_state = flax.core.freeze({})
        else:
            model_state = flax.core.freeze(model_state)

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            model_state=model_state,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params, **self.model_state}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def build_checkpoint_metadata(
    config,
    *,
    environment,
    seed,
    git_commit,
    code_signature=None,
    resume=False,
    restore_path=None,
):
    """Build the architecture signature stored beside and inside checkpoints.

    The signature intentionally contains only fields that must agree when a
    checkpoint is restored.  Runtime provenance such as ``resume`` is kept in
    the surrounding metadata, but is not part of the equality check.
    """

    action_core_type = config.get('action_core_type')
    action_block_type = config.get('action_block_type')
    if action_block_type in (None, '', 'auto', 'legacy'):
        action_block_type = config.get('hrm_block_type')
    if action_block_type in (None, '', 'auto', 'legacy'):
        action_block_type = (
            'swiglu_rmsnorm'
            if config.get('action_hrm_is_mlp_mixer', False)
            else 'attention'
        )
    if action_core_type == 'vanilla_coghp' and action_block_type in (
        None,
        '',
        'auto',
        'legacy',
        'swiglu_rmsnorm',
    ):
        action_block_type = 'coghp_gelu'

    mixer_intermediate_size = config.get(
        'hrm_mixer_intermediate_size',
        -1,
    )
    if mixer_intermediate_size == -1:
        mixer_intermediate_size = config.get(
            'action_hrm_mlp_mixer_intermediate_size'
        )

    is_nonrecurrent_action = action_core_type in (
        'vanilla_coghp',
        'matched_feedforward',
    )
    planner_core_type = config.get('planner_core_type', 'vanilla_coghp')
    if planner_core_type == 'legacy_coghp':
        planner_core_type = 'vanilla_coghp'
    planner_is_vanilla = planner_core_type == 'vanilla_coghp'
    planner_is_matched = planner_core_type == 'matched_feedforward'
    planner_is_hrm = planner_core_type == 'hrm'
    share_actor_core = bool(config.get('share_actor_core', False))
    signature = {
        'agent_name': config.get('agent_name'),
        'action_core_type': action_core_type,
        'action_block_type': action_block_type,
        'share_actor_core': share_actor_core,
        'mixer_intermediate_size': mixer_intermediate_size,
        'num_layers': config.get('action_hrm_num_layers'),
        'feature_dim': config.get('feature_dim'),
        'num_subgoals': config.get('num_subgoals'),
        'environment': environment,
        'seed': seed,
        'git_commit': git_commit,
        'code_signature': code_signature,
        'outer_cycles': 1,
        'h_cycles': (
            'N/A'
            if is_nonrecurrent_action
            else (
                config.get('planner_h_cycles')
                if share_actor_core
                else config.get('action_hrm_h_cycles')
            )
        ),
        'l_cycles': (
            'N/A'
            if is_nonrecurrent_action
            else (
                config.get('planner_l_cycles')
                if share_actor_core
                else config.get('action_hrm_l_cycles')
            )
        ),
        'gradient_mode': (
            'N/A'
            if is_nonrecurrent_action
            else (
                'full_bptt'
                if (
                    config.get('planner_bptt', True)
                    if share_actor_core
                    else config.get('action_hrm_bptt', False)
                )
                else 'stage_truncated'
            )
        ),
        # Planner controls are included in schema v2.  For the legacy vanilla
        # and matched paths these do not alter the old execution semantics,
        # while an official-HRM planner checkpoint is explicitly identifiable.
        'planner_core_type': planner_core_type,
        'planner_block_type': config.get(
            'planner_block_type',
            'swiglu_rmsnorm',
        ),
        'planner_num_layers': config.get('planner_num_layers', 1),
        'planner_mixer_intermediate_size': config.get(
            'planner_mixer_intermediate_size',
            256,
        ),
        'planner_h_cycles': (
            'N/A'
            if planner_is_vanilla or planner_is_matched
            else config.get('planner_h_cycles', 2)
        ),
        'planner_l_cycles': (
            'N/A'
            if planner_is_vanilla or planner_is_matched
            else config.get('planner_l_cycles', 2)
        ),
        'planner_gradient_mode': (
            'N/A'
            if not planner_is_hrm
            else (
                'full_bptt'
                if config.get('planner_bptt', True)
                else 'stage_truncated'
            )
        ),
        'planner_executor_shared_core': share_actor_core,
    }
    if config.get('agent_name') == 'navigation_coghp':
        for key in ('command_content', 'high_weight_source', 'navigation_protocol',
                    'position_scale', 'position_mean', 'position_hidden_dims',
                    'position_loss_coef', 'global_dim', 'global_hidden_dims',
                    'global_abstraction_steps', 'global_discount', 'global_goal_tolerance',
                    'awr_max_weight'):
            value = config.get(key)
            signature[key] = list(value) if isinstance(value, (tuple, list)) else value
    return {
        'schema_version': 2,
        'signature': signature,
        'resume': bool(resume),
        'restore_path': restore_path,
    }


def validate_checkpoint_metadata(
    loaded_metadata,
    expected_metadata,
    *,
    require_metadata=False,
):
    """Reject a checkpoint whose recorded architecture signature differs."""

    if loaded_metadata is None:
        if require_metadata:
            raise ValueError('Checkpoint metadata is required but missing.')
        return
    if expected_metadata is None:
        return

    loaded_signature = loaded_metadata.get('signature')
    expected_signature = expected_metadata.get('signature')
    if loaded_signature is None or expected_signature is None:
        if require_metadata:
            raise ValueError('Checkpoint signature is required but missing.')
        return
    # Schema-1 checkpoints predate planner selectors.  They remain restorable
    # for the old vanilla/matched paths, but cannot be used to resume an
    # official-HRM planner because its recurrence is not represented in the
    # old signature.
    loaded_schema = int(loaded_metadata.get('schema_version', 1))
    # Schema-2 checkpoints written before the shared actor-core option did
    # not record these two fields.  Treat their absence as the historical
    # split topology; a shared run (expected=True) still fails closed below.
    if loaded_schema >= 2:
        loaded_signature = dict(loaded_signature)
        loaded_signature.setdefault('share_actor_core', False)
        loaded_signature.setdefault('planner_executor_shared_core', False)
    if loaded_schema < 2:
        if expected_signature.get('planner_core_type') == 'hrm':
            raise ValueError(
                'Checkpoint metadata schema v2 is required for an official '
                'HRM planner.'
            )
        planner_keys = {
            'planner_core_type',
            'planner_block_type',
            'planner_num_layers',
            'planner_mixer_intermediate_size',
            'planner_h_cycles',
            'planner_l_cycles',
            'planner_gradient_mode',
            'share_actor_core',
            'planner_executor_shared_core',
        }
        loaded_signature = {
            key: value
            for key, value in loaded_signature.items()
            if key not in planner_keys
        }
        expected_signature = {
            key: value
            for key, value in expected_signature.items()
            if key not in planner_keys
        }

    if loaded_signature != expected_signature:
        differing_keys = sorted(
            key
            for key in set(loaded_signature) | set(expected_signature)
            if loaded_signature.get(key) != expected_signature.get(key)
        )
        differences = {
            key: {
                'checkpoint': loaded_signature.get(key),
                'expected': expected_signature.get(key),
            }
            for key in differing_keys
        }
        raise ValueError(
            'Checkpoint configuration signature mismatch: '
            + json.dumps(differences, sort_keys=True)
        )


def save_agent(agent, save_dir, epoch, checkpoint_metadata=None):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
        checkpoint_metadata=checkpoint_metadata,
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(
    agent,
    restore_path,
    restore_epoch,
    expected_checkpoint_metadata=None,
    require_checkpoint_metadata=False,
):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    validate_checkpoint_metadata(
        load_dict.get('checkpoint_metadata'),
        expected_checkpoint_metadata,
        require_metadata=require_checkpoint_metadata,
    )

    loaded_agent_state = load_dict['agent']
    current_agent_state = flax.serialization.to_state_dict(agent)
    # Checkpoints created before TrainState stored non-parameter collections do
    # not contain ``model_state``.  Empty/current collections are the only
    # sensible fallback for those legacy checkpoints.
    if (
        isinstance(loaded_agent_state, Mapping)
        and isinstance(current_agent_state, Mapping)
        and 'network' in loaded_agent_state
        and 'network' in current_agent_state
        and 'model_state' not in loaded_agent_state['network']
    ):
        loaded_agent_state['network']['model_state'] = (
            current_agent_state['network']['model_state']
        )

    agent = flax.serialization.from_state_dict(agent, loaded_agent_state)

    print(f'Restored from {restore_path}')

    return agent
