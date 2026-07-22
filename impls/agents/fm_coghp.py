"""Flow-Matching CoGHP agent with pluggable candidate selectors."""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.coghp import CoGHPAgent, get_config as get_coghp_config
from utils.bridge_critics import LocalCRLBridgeCritic
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.fm_coghp_network import FlowChainPlanner
from utils.coghp_network import HierarchicalPolicyNetwork
from utils.networks import GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, MLP


class FMCoGHPAgent(CoGHPAgent):
    """CoGHP with joint flow generation of the complete latent goal chain."""

    @staticmethod
    def normalize_chain(chain):
        # Match utils.networks.LengthNormalize, whose output radius is sqrt(D).
        radius = jnp.sqrt(jnp.asarray(chain.shape[-1], dtype=chain.dtype))
        return chain / (jnp.linalg.norm(chain, axis=-1, keepdims=True) + 1e-6) * radius

    @staticmethod
    def nearest_subgoal(chain):
        """Return the nearest slot; CoGHP chains are ordered far-to-near."""
        return chain[..., -1, :]

    def flow_target_reps(self, batch):
        observations = batch['observations']
        targets = batch['high_actor_targets']
        obs_expand = jnp.repeat(
            observations[:, None, ...],
            repeats=self.config['num_subgoals'],
            axis=1,
        )
        target_reps = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, targets], axis=-1),
        )
        return jax.lax.stop_gradient(target_reps)

    def sample_source_chain(self, rng, shape):
        if self.config.get('flow_noise_type', 'unit_gaussian') != 'unit_gaussian':
            raise ValueError('FM-CoGHP V1 supports only flow_noise_type="unit_gaussian".')
        return self.normalize_chain(jax.random.normal(rng, shape))

    def flow_loss(self, batch, grad_params, rng):
        noise_rng, time_rng = jax.random.split(rng)
        target_chain = self.flow_target_reps(batch)
        source_chain = self.sample_source_chain(noise_rng, target_chain.shape)
        tau = jax.random.uniform(
            time_rng,
            shape=(target_chain.shape[0], 1, 1),
            minval=self.config['flow_time_eps'],
            maxval=1.0 - self.config['flow_time_eps'],
        )
        chain_t = (1.0 - tau) * source_chain + tau * target_chain
        target_velocity = target_chain - source_chain
        pred_velocity = self.network.select('flow_planner')(
            batch['observations'],
            batch['high_actor_goals'],
            chain_t,
            tau[:, 0, 0],
            params=grad_params,
        )

        slot_velocity_mse = jnp.mean(jnp.square(pred_velocity - target_velocity), axis=-1)
        pred_clean = chain_t + (1.0 - tau) * pred_velocity
        slot_clean_mse = jnp.mean(jnp.square(pred_clean - target_chain), axis=-1)
        velocity_loss = jnp.mean(slot_velocity_mse)
        clean_loss = jnp.mean(slot_clean_mse)
        loss = velocity_loss + self.config['flow_clean_loss_weight'] * clean_loss

        return loss, {
            'loss': loss,
            'velocity_mse': velocity_loss,
            'clean_mse': clean_loss,
            'nearest_velocity_mse': jnp.mean(slot_velocity_mse[:, -1]),
            'nearest_clean_mse': jnp.mean(slot_clean_mse[:, -1]),
            'farthest_clean_mse': jnp.mean(slot_clean_mse[:, 0]),
            'source_norm': jnp.mean(jnp.linalg.norm(source_chain, axis=-1)),
            'target_norm': jnp.mean(jnp.linalg.norm(target_chain, axis=-1)),
            'pred_clean_norm': jnp.mean(jnp.linalg.norm(pred_clean, axis=-1)),
        }

    def _crl_false_negative_mask(self, batch):
        batch_size = batch['observations'].shape[0]
        if not self.config.get('crl_mask_false_negatives', True) or 'indices' not in batch:
            return jnp.zeros((batch_size, batch_size), dtype=bool)

        query_idxs = batch['indices'][:, None]
        query_final_idxs = batch['trajectory_final_idxs'][:, None]
        goal_idxs = batch['high_actor_target_idxs'][:, -1][None, :]
        goal_final_idxs = batch['high_actor_target_final_idxs'][:, -1][None, :]
        temporal_gap = goal_idxs - query_idxs
        mask = (
            (query_final_idxs == goal_final_idxs)
            & (temporal_gap > 0)
            & (
                temporal_gap
                <= self.config['crl_horizon'] + self.config['crl_false_negative_margin']
            )
        )
        return mask.at[jnp.arange(batch_size), jnp.arange(batch_size)].set(False)

    def crl_loss(self, batch, grad_params):
        positive_goal_reps = self.nearest_subgoal(self.flow_target_reps(batch))
        logits = self.network.select('crl_critic')(
            batch['observations'],
            positive_goal_reps,
            pairwise=True,
            params=grad_params,
        )
        false_negative_mask = self._crl_false_negative_mask(batch)
        masked_logits = jnp.where(false_negative_mask, -jnp.inf, logits)
        labels = jnp.arange(logits.shape[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(masked_logits, labels).mean()

        positive_logits = jnp.diag(logits)
        negative_mask = ~jnp.eye(logits.shape[0], dtype=bool) & ~false_negative_mask
        negative_count = jnp.maximum(jnp.sum(negative_mask), 1)
        negative_mean = jnp.sum(jnp.where(negative_mask, logits, 0.0)) / negative_count
        row_negative_max = jnp.max(
            jnp.where(negative_mask, logits, -jnp.inf),
            axis=1,
        )
        valid_negative_rows = jnp.any(negative_mask, axis=1)
        score_margin = jnp.mean(
            jnp.where(valid_negative_rows, positive_logits - row_negative_max, 0.0)
        )
        top1_accuracy = jnp.mean(jnp.argmax(masked_logits, axis=1) == labels)
        ranking_auc = jnp.sum(
            jnp.where(negative_mask, positive_logits[:, None] > logits, 0.0)
        ) / negative_count

        return loss, {
            'loss': loss,
            'positive_logit': jnp.mean(positive_logits),
            'negative_logit': negative_mean,
            'score_margin': score_margin,
            'top1_accuracy': top1_accuracy,
            'ranking_auc': ranking_auc,
            'potential_false_negative_ratio': jnp.mean(false_negative_mask.astype(jnp.float32)),
        }

    def actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        obs_expand = jnp.repeat(
            observations[:, None, ...],
            repeats=self.config['num_subgoals'],
            axis=1,
        )
        subgoal_reps = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1),
            params=grad_params,
        )
        high_dist, low_dist, _ = self.network.select('actor_mixer')(
            observations,
            batch['high_actor_goals'],
            rng,
            subgoal_reps=subgoal_reps,
            action_seq=batch['actions'],
            params=grad_params,
        )
        low_loss, low_info = self.low_actor_loss(batch, low_dist)
        high_info = {}
        if self.config['flow_high_aux_weight'] > 0.0:
            high_loss, high_info = self.multi_high_actor_loss(
                batch,
                high_dist,
                obs_expand,
                grad_params,
            )
        else:
            high_loss = 0.0
        actor_loss = low_loss + self.config['flow_high_aux_weight'] * high_loss
        high_info['aux_weight'] = jnp.asarray(self.config['flow_high_aux_weight'])
        return actor_loss, high_info, low_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        flow_rng, actor_rng = jax.random.split(rng)
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        actor_loss, high_actor_info, low_actor_info = self.actor_loss(batch, grad_params, actor_rng)
        flow_loss, flow_info = self.flow_loss(batch, grad_params, flow_rng)
        crl_loss, crl_info = self.crl_loss(batch, grad_params)

        for key, value in value_info.items():
            info[f'value/{key}'] = value
        for key, value in high_actor_info.items():
            info[f'high_actor/{key}'] = value
        for key, value in low_actor_info.items():
            info[f'low_actor/{key}'] = value
        for key, value in flow_info.items():
            info[f'flow/{key}'] = value
        for key, value in crl_info.items():
            info[f'crl/{key}'] = value

        loss = (
            value_loss
            + actor_loss
            + self.config['flow_loss_weight'] * flow_loss
            + self.config['crl_loss_weight'] * crl_loss
        )
        info['total_loss'] = loss
        return loss, info

    def sample_flow_chains(self, observations, goals, rng, num_candidates=None, num_steps=None):
        if goals is None:
            raise ValueError('FM-CoGHP requires final goals when sampling chains.')
        num_candidates = num_candidates or self.config['flow_num_candidates']
        num_steps = num_steps or self.config['flow_steps']
        batch_size = observations.shape[0]
        num_subgoals = self.config['num_subgoals']
        feature_dim = self.config['feature_dim']

        obs_repeat = jnp.repeat(observations[:, None, ...], num_candidates, axis=1).reshape(
            batch_size * num_candidates, *observations.shape[1:]
        )
        goal_repeat = jnp.repeat(goals[:, None, ...], num_candidates, axis=1).reshape(
            batch_size * num_candidates, *goals.shape[1:]
        )
        chain = self.sample_source_chain(
            rng,
            (batch_size * num_candidates, num_subgoals, feature_dim),
        )

        def body_fn(step, current_chain):
            tau = step.astype(current_chain.dtype) / jnp.asarray(num_steps, dtype=current_chain.dtype)
            tau_batch = jnp.full((batch_size * num_candidates,), tau, dtype=current_chain.dtype)
            velocity = self.network.select('flow_planner')(
                obs_repeat,
                goal_repeat,
                current_chain,
                tau_batch,
            )
            return current_chain + velocity / jnp.asarray(num_steps, dtype=current_chain.dtype)

        chain = jax.lax.fori_loop(0, num_steps, body_fn, chain)
        pre_normalize_chain = chain
        if self.config['flow_normalize_final']:
            chain = self.normalize_chain(chain)
        chains = chain.reshape(batch_size, num_candidates, num_subgoals, feature_dim)
        pre_normalize_chains = pre_normalize_chain.reshape(
            batch_size, num_candidates, num_subgoals, feature_dim
        )
        return chains, pre_normalize_chains

    def score_flow_candidates(self, observations, candidate_chains, selector):
        nearest_goal_reps = self.nearest_subgoal(candidate_chains)
        batch_size, num_candidates = nearest_goal_reps.shape[:2]
        if selector in ('none', 'first'):
            return jnp.zeros((batch_size, num_candidates), dtype=nearest_goal_reps.dtype)
        if selector == 'coghp_value':
            obs_flat = jnp.repeat(observations[:, None, ...], num_candidates, axis=1).reshape(
                batch_size * num_candidates, *observations.shape[1:]
            )
            goal_flat = nearest_goal_reps.reshape(batch_size * num_candidates, -1)
            v1, v2 = self.network.select('value')(
                obs_flat,
                goal_flat,
                goal_encoded=True,
            )
            v1 = v1.reshape(batch_size, num_candidates)
            v2 = v2.reshape(batch_size, num_candidates)
            reduce = self.config.get('flow_value_selector_reduce', 'min')
            if reduce == 'min':
                return jnp.minimum(v1, v2)
            if reduce == 'mean':
                return (v1 + v2) / 2.0
            raise ValueError(f'Unknown flow_value_selector_reduce: {reduce}')
        if selector == 'local_crl':
            return self.network.select('crl_critic')(observations, nearest_goal_reps)
        raise ValueError(f'Unknown Flow selector: {selector}')

    def select_flow_chain(self, observations, candidate_chains, selector):
        scores = self.score_flow_candidates(observations, candidate_chains, selector)
        if selector in ('none', 'first'):
            selected_idx = jnp.zeros((candidate_chains.shape[0],), dtype=jnp.int32)
        else:
            selected_idx = jnp.argmax(scores, axis=1)
        selected_chain = candidate_chains[jnp.arange(candidate_chains.shape[0]), selected_idx]
        return selected_chain, scores, selected_idx

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        seed = self.rng if seed is None else seed
        flow_rng, action_rng = jax.random.split(seed)
        observations = jnp.expand_dims(observations, axis=0)
        goals = jnp.expand_dims(goals, axis=0) if goals is not None else None
        chains, _ = self.sample_flow_chains(observations, goals, flow_rng)
        selected_chain, _, _ = self.select_flow_chain(
            observations,
            chains,
            self.config['flow_selector'],
        )
        if self.config.get('stop_action_gradient_to_flow', True):
            selected_chain = jax.lax.stop_gradient(selected_chain)
        _, _, predicted_actions = self.network.select('actor_mixer')(
            observations,
            goals,
            action_rng,
            subgoal_reps=selected_chain,
            action_seq=None,
            temperature=temperature,
        )
        actions = predicted_actions[0]
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1.0, 1.0)
        return actions

    def _action_mse_for_chain(self, batch, chain, rng):
        _, low_dist, _ = self.network.select('actor_mixer')(
            batch['observations'],
            batch['high_actor_goals'],
            rng,
            subgoal_reps=chain,
            action_seq=None,
            temperature=1.0,
        )
        return jnp.mean(jnp.square(low_dist.mode() - batch['actions']))

    def validation_rollout_info(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        flow_rng, action_rng = jax.random.split(rng)
        target_chain = self.flow_target_reps(batch)
        candidates, pre_normalize = self.sample_flow_chains(
            batch['observations'],
            batch['high_actor_goals'],
            flow_rng,
            num_candidates=self.config['flow_diagnostic_candidates'],
        )
        target_nearest = self.nearest_subgoal(target_chain)[:, None, :]
        nearest_mse = jnp.mean(
            jnp.square(self.nearest_subgoal(candidates) - target_nearest),
            axis=-1,
        )
        full_chain_mse = jnp.mean(
            jnp.square(candidates - target_chain[:, None, :, :]),
            axis=(-1, -2),
        )
        oracle_idx = jnp.argmin(nearest_mse, axis=1)
        first_idx = jnp.zeros_like(oracle_idx)
        value_scores = self.score_flow_candidates(batch['observations'], candidates, 'coghp_value')
        crl_scores = self.score_flow_candidates(batch['observations'], candidates, 'local_crl')
        value_idx = jnp.argmax(value_scores, axis=1)
        crl_idx = jnp.argmax(crl_scores, axis=1)

        def gather_idx(values, idxs):
            return values[jnp.arange(values.shape[0]), idxs]

        def gather_chain(idxs):
            return candidates[jnp.arange(candidates.shape[0]), idxs]

        def mean_rank(idxs):
            selected_errors = gather_idx(nearest_mse, idxs)
            return jnp.mean(1 + jnp.sum(nearest_mse < selected_errors[:, None], axis=1))

        info = {
            'fm_selector/first_nearest_mse': jnp.mean(gather_idx(nearest_mse, first_idx)),
            'fm_selector/value_nearest_mse': jnp.mean(gather_idx(nearest_mse, value_idx)),
            'fm_selector/crl_nearest_mse': jnp.mean(gather_idx(nearest_mse, crl_idx)),
            'fm_selector/oracle_nearest_mse': jnp.mean(gather_idx(nearest_mse, oracle_idx)),
            'fm_selector/first_full_chain_mse': jnp.mean(gather_idx(full_chain_mse, first_idx)),
            'fm_selector/value_full_chain_mse': jnp.mean(gather_idx(full_chain_mse, value_idx)),
            'fm_selector/crl_full_chain_mse': jnp.mean(gather_idx(full_chain_mse, crl_idx)),
            'fm_selector/oracle_full_chain_mse': jnp.mean(gather_idx(full_chain_mse, oracle_idx)),
            'fm_selector/value_oracle_agreement': jnp.mean(value_idx == oracle_idx),
            'fm_selector/crl_oracle_agreement': jnp.mean(crl_idx == oracle_idx),
            'fm_selector/value_score_std': jnp.std(value_scores),
            'fm_selector/crl_score_std': jnp.std(crl_scores),
            'fm_selector/value_selected_rank': mean_rank(value_idx),
            'fm_selector/crl_selected_rank': mean_rank(crl_idx),
            'flow/best_of_m_nearest_mse': jnp.mean(jnp.min(nearest_mse, axis=1)),
            'flow/mean_candidate_nearest_mse': jnp.mean(nearest_mse),
            'flow/nearest_candidate_std': jnp.mean(jnp.std(self.nearest_subgoal(candidates), axis=1)),
            'flow/farthest_candidate_std': jnp.mean(jnp.std(candidates[:, :, 0, :], axis=1)),
            'flow/final_norm_before_normalize': jnp.mean(jnp.linalg.norm(pre_normalize, axis=-1)),
            'flow/final_norm_after_normalize': jnp.mean(jnp.linalg.norm(candidates, axis=-1)),
        }

        num_candidates = candidates.shape[1]
        if num_candidates > 1:
            differences = candidates[:, :, None, :, :] - candidates[:, None, :, :, :]
            pairwise_distances = jnp.sqrt(jnp.mean(jnp.square(differences), axis=(-1, -2)) + 1e-12)
            pairwise_mask = ~jnp.eye(num_candidates, dtype=bool)[None, :, :]
            info['flow/sample_pairwise_distance'] = jnp.sum(
                jnp.where(pairwise_mask, pairwise_distances, 0.0)
            ) / (candidates.shape[0] * num_candidates * (num_candidates - 1))
        else:
            info['flow/sample_pairwise_distance'] = jnp.array(0.0)

        if self.config.get('flow_diagnostic_action_mse', True):
            action_keys = jax.random.split(action_rng, 4)
            info.update(
                {
                    'fm_selector/first_action_mse': self._action_mse_for_chain(
                        batch, gather_chain(first_idx), action_keys[0]
                    ),
                    'fm_selector/value_action_mse': self._action_mse_for_chain(
                        batch, gather_chain(value_idx), action_keys[1]
                    ),
                    'fm_selector/crl_action_mse': self._action_mse_for_chain(
                        batch, gather_chain(crl_idx), action_keys[2]
                    ),
                    'fm_selector/oracle_action_mse': self._action_mse_for_chain(
                        batch, gather_chain(oracle_idx), action_keys[3]
                    ),
                }
            )
        return info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config['num_subgoals'] < 1:
            raise ValueError('FM-CoGHP requires num_subgoals >= 1.')
        if not config.get('stop_action_gradient_to_flow', True):
            raise ValueError('FM-CoGHP V1 requires stop_action_gradient_to_flow=True.')
        if not config.get('stop_critic_gradient_to_flow', True):
            raise ValueError('FM-CoGHP V1 requires stop_critic_gradient_to_flow=True.')
        if config.get('flow_training_mode', 'joint') != 'joint':
            raise ValueError('FM-CoGHP V1 currently supports only flow_training_mode="joint".')
        if config['flow_selector'] not in ('none', 'first', 'coghp_value', 'local_crl'):
            raise ValueError(f"Unknown Flow selector: {config['flow_selector']}")
        if config['flow_steps'] < 1 or config['flow_num_candidates'] < 1:
            raise ValueError('flow_steps and flow_num_candidates must both be positive.')
        if not 0.0 <= config['flow_time_eps'] < 0.5:
            raise ValueError('flow_time_eps must be in [0, 0.5).')
        if config['crl_temperature'] <= 0.0:
            raise ValueError('crl_temperature must be positive.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        ex_goals = ex_observations
        if config['discrete']:
            action_dim = jnp.max(ex_actions) + 1
        else:
            action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            encoder_module = None
            goal_rep_seq = []
        goal_rep_seq.extend(
            [
                MLP(
                    hidden_dims=(*config['enc_hidden_dims'], config['feature_dim']),
                    activate_final=False,
                    layer_norm=config['layer_norm'],
                ),
                LengthNormalize(),
            ]
        )
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if encoder_module is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
            crl_state_encoder_def = encoder_module()
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None
            crl_state_encoder_def = Identity()

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
            raise NotImplementedError('Discrete actions are not supported by FM-CoGHP V1.')
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
        gc_enc = low_actor_encoder_def if config['gc_enc'] == 'concat' else high_actor_encoder_def
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
        flow_planner_def = FlowChainPlanner(
            num_subgoals=config['num_subgoals'],
            feature_dim=config['feature_dim'],
            num_mixer_blocks=config['flow_num_blocks'],
            mixer_token_hidden=config['flow_token_hidden'],
            mixer_channel_hidden=config['flow_channel_hidden'],
            flow_time_embed_dim=config['flow_time_embed_dim'],
            gc_encoder=gc_enc,
            enc_hidden=config['enc_hidden_dims'],
            layer_norm=config['layer_norm'],
        )
        crl_critic_def = LocalCRLBridgeCritic(
            latent_dim=config['crl_latent_dim'],
            temperature=config['crl_temperature'],
            hidden_dims=config['crl_hidden_dims'],
            state_encoder=crl_state_encoder_def,
            layer_norm=config['layer_norm'],
        )

        batch_size = ex_observations.shape[0]
        ex_chain = jnp.zeros(
            (batch_size, config['num_subgoals'], config['feature_dim']),
            dtype=jnp.float32,
        )
        ex_flow_times = jnp.zeros((batch_size,), dtype=jnp.float32)
        ex_goal_reps = jnp.zeros((batch_size, config['feature_dim']), dtype=jnp.float32)
        network_info = {
            'goal_rep': (goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1),)),
            'value': (value_def, (ex_observations, ex_goals)),
            'target_value': (target_value_def, (ex_observations, ex_goals)),
            'actor_mixer': (actor_mixer_def, (ex_observations, ex_goals, rng)),
            'flow_planner': (
                flow_planner_def,
                (ex_observations, ex_goals, ex_chain, ex_flow_times),
            ),
            'crl_critic': (crl_critic_def, (ex_observations, ex_goal_reps)),
        }
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_args = {key: value[1] for key, value in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_value'] = network_params['modules_value']
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        print('Creating FM-CoGHP Done')
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_coghp_config()
    config.agent_name = 'fm_coghp'
    config.flow_num_blocks = 2
    config.flow_token_hidden = 64
    config.flow_channel_hidden = 128
    config.flow_time_embed_dim = 64
    config.flow_steps = 8
    config.flow_num_candidates = 4
    config.flow_selector = 'none'
    config.flow_noise_type = 'unit_gaussian'
    config.flow_normalize_final = True
    config.flow_time_eps = 1e-4
    config.flow_loss_weight = 1.0
    config.flow_clean_loss_weight = 0.1
    config.flow_high_aux_weight = 0.0
    config.flow_value_selector_reduce = 'min'
    config.crl_loss_weight = 1.0
    config.crl_latent_dim = 128
    config.crl_hidden_dims = (256, 256)
    config.crl_temperature = 0.1
    config.crl_horizon = 50
    config.crl_false_negative_margin = 10
    config.crl_mask_false_negatives = True
    config.enable_fm_diagnostics = True
    config.flow_diagnostic_candidates = 4
    config.flow_diagnostic_action_mse = True
    config.stop_action_gradient_to_flow = True
    config.stop_critic_gradient_to_flow = True
    config.flow_training_mode = 'joint'
    return config
