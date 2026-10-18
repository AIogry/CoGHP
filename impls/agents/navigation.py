"""Independent hierarchical navigation learner.

CoGHP remains a separate baseline. This learner owns its value/AWR objectives,
command representations and planner/executor construction. Historical profiles
preserve existing experiment semantics; canonical defaults use native backbones.
"""

import copy
from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from networks.backbones import SwiGLUBackbone
from networks.policies import NavigationPolicy
from utils.encoders import GCEncoder, PositionCommand, PositionDecoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor,
    GCValue,
    GlobalPotential,
    Identity,
    LengthNormalize,
    MLP,
)


def stable_awr(advantage, beta, cap=100.0):
    return jax.lax.stop_gradient(jnp.exp(jnp.minimum(beta * advantage, jnp.log(cap))))


def global_ivl_loss(
    values, current_targets, next_targets, rewards, masks, valid, discount, expectile
):
    """Exact legacy IVL twin-head/sign convention, with masked OTA labels."""
    sign_target = rewards + discount * masks * jnp.min(next_targets, axis=0)
    advantage = jax.lax.stop_gradient(sign_target - jnp.mean(current_targets, axis=0))
    targets = jax.lax.stop_gradient(
        rewards[None] + discount * masks[None] * next_targets
    )
    weights = jnp.where(advantage >= 0, expectile, 1.0 - expectile)
    errors = weights[None] * (targets - values) ** 2
    return (errors * valid[None]).sum() / jnp.maximum(valid.sum(), 1.0)


class NavigationAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    all_time_actions: Any = None
    goal_temp: Any = None

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, 1 - expectile)
        return weight * diff**2

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        next_v1_t, next_v2_t = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t
        v1_t, v2_t = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t
        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2
        return (
            value_loss,
            {
                "value_loss": value_loss,
                "v_mean": v.mean(),
                "v_max": v.max(),
                "v_min": v.min(),
            },
        )

    def _detail_high_actor_loss(self, batch, dist_list, obs_expand, grad_params):
        multi_targets = self.network.select("goal_rep")(
            jnp.concatenate([obs_expand, batch["high_actor_targets"]], axis=-1)
        )
        actor_loss, adv_mean, bc_log_prob, mse, std = (0, 0, 0, 0, 0)
        for i in range(self.config["num_subgoals"]):
            v1, v2 = self.network.select("value")(
                batch["observations"], batch["high_actor_goals"]
            )
            nv1, nv2 = self.network.select("value")(
                batch["high_actor_targets"][:, i, :], batch["high_actor_goals"]
            )
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2
            adv = nv - v
            exp_a = jnp.exp(adv * self.config["high_alpha"])
            exp_a = jnp.minimum(exp_a, 100.0)
            target = multi_targets[:, i, :]
            log_prob = dist_list[i].log_prob(target)
            actor_loss += -(
                (exp_a * log_prob).mean() / self.config["subgoal_steps"]
            ) * self.config["high_discount"] ** (self.config["num_subgoals"] - i - 1)
            adv_mean += adv.mean()
            bc_log_prob += log_prob.mean()
            mse += jnp.mean((dist_list[i].mode() - target) ** 2)
            std += jnp.mean(dist_list[i].scale_diag)
        return (
            actor_loss / self.config["num_subgoals"],
            {
                "actor_loss": actor_loss,
                "adv": adv_mean / self.config["num_subgoals"],
                "bc_log_prob": bc_log_prob / self.config["num_subgoals"],
                "mse": mse / self.config["num_subgoals"],
                "std": std / self.config["num_subgoals"],
            },
        )

    def low_actor_loss(self, batch, dist):
        if self.config["num_subgoals"] > 0:
            target = batch["high_actor_targets"][:, -1, :]
        else:
            target = batch["high_actor_goals"]
        v1, v2 = self.network.select("value")(batch["observations"], target)
        nv1, nv2 = self.network.select("value")(batch["next_observations"], target)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        exp_a = jnp.exp(adv * self.config["low_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)
        action = batch["actions"]
        log_prob = dist.log_prob(action)
        actor_loss = -(exp_a * log_prob).mean()
        actor_info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - action) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )
        return (actor_loss, actor_info)

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def _full_state_actor_loss(self, batch, grad_params, rng):
        observations = batch["observations"]
        goals = batch["high_actor_goals"]
        observation_sequence = jnp.repeat(
            observations[:, None, :], self.config["num_subgoals"], axis=1
        )
        teacher_subgoals = self.network.select("goal_rep")(
            jnp.concatenate(
                [observation_sequence, batch["high_actor_targets"]], axis=-1
            ),
            params=grad_params,
        )
        high_dists, action_dist, _, diagnostics = self.network.select("actor_mixer")(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            action_seq=batch["actions"],
            params=grad_params,
            return_diagnostics=True,
        )
        high_actor_loss, high_actor_info = self.multi_high_actor_loss(
            batch, high_dists, observation_sequence, grad_params
        )
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, action_dist)
        return (
            high_actor_loss + low_actor_loss,
            high_actor_info,
            low_actor_info,
            diagnostics,
        )

    @jax.jit
    def _actor_value_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        for key, value in value_info.items():
            info[f"value/{key}"] = value
        actor_loss, high_info, low_info, diagnostics = self.actor_loss(
            batch, grad_params, rng
        )
        for key, value in high_info.items():
            info[f"high_actor/{key}"] = value
        for key, value in low_info.items():
            info[f"low_actor/{key}"] = value
        info.update(diagnostics)
        return (value_loss + actor_loss, info)

    def _full_state_validation_info(self, batch, rng=None):
        """Measure the action error under teacher and free-running chains."""
        rng = self.rng if rng is None else rng
        observations = batch["observations"]
        goals = batch["high_actor_goals"]
        observation_sequence = jnp.repeat(
            observations[:, None, :], self.config["num_subgoals"], axis=1
        )
        teacher_subgoals = self.network.select("goal_rep")(
            jnp.concatenate(
                [observation_sequence, batch["high_actor_targets"]], axis=-1
            )
        )
        _, teacher_dist, _ = self.network.select("actor_mixer")(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=0.0,
            action_chain_source="teacher",
        )
        _, predicted_dist, _ = self.network.select("actor_mixer")(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=0.0,
            action_chain_source="predicted",
        )
        _, teacher_density_dist, _ = self.network.select("actor_mixer")(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=1.0,
            action_chain_source="teacher",
        )
        _, predicted_density_dist, _ = self.network.select("actor_mixer")(
            observations,
            goals,
            rng,
            subgoal_reps=teacher_subgoals,
            temperature=1.0,
            action_chain_source="predicted",
        )
        actions = batch["actions"]
        teacher_mse = jnp.mean((teacher_dist.mode() - actions) ** 2)
        predicted_mse = jnp.mean((predicted_dist.mode() - actions) ** 2)
        return {
            "validation_action/teacher/mse": teacher_mse,
            "validation_action/predicted/mse": predicted_mse,
            "validation_action/gap/mse": predicted_mse - teacher_mse,
            "validation_action/teacher/bc_log_prob": teacher_density_dist.log_prob(
                actions
            ).mean(),
            "validation_action/predicted/bc_log_prob": predicted_density_dist.log_prob(
                actions
            ).mean(),
        }

    @jax.jit
    def _full_state_actions(self, observations, goals=None, seed=None, temperature=1.0):
        observations = observations[None, :]
        if goals is not None:
            goals = goals[None, :]
        _, _, predicted_actions = self.network.select("actor_mixer")(
            observations,
            goals,
            seed,
            subgoal_reps=None,
            action_seq=None,
            temperature=temperature,
        )
        actions = predicted_actions[0]
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def _create_core(cls, seed, ex_observations, ex_actions, config):
        if config["num_subgoals"] < 1:
            raise ValueError("Navigation requires num_subgoals >= 1.")
        if config["train_action_chain_source"] not in ("teacher", "predicted"):
            raise ValueError("train_action_chain_source must be teacher or predicted.")
        if config["action_hrm_forward_dtype"] not in ("float32", "bfloat16"):
            raise ValueError("action_hrm_forward_dtype must be float32 or bfloat16.")
        if config["discrete"]:
            raise NotImplementedError("Discrete actions are not supported.")
        network_backbone = config.get("network_backbone", "mlp")
        if network_backbone not in ("mlp", "swiglu_rmsnorm"):
            raise ValueError("Unknown representation/value backbone.")
        native = network_backbone == "swiglu_rmsnorm"
        if native and (
            config.get("planner_core_type") not in ("matched_feedforward", "hrm")
            or config.get("action_core_type") == "vanilla_coghp"
            or config.get("planner_block_type") != "swiglu_rmsnorm"
            or (config.get("action_block_type") != "swiglu_rmsnorm")
        ):
            raise ValueError(
                "Native navigation uses independent SwiGLU/RMSNorm backbones; use an explicit legacy profile for historical Mixer controls."
            )
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            goal_rep_modules = [encoder_module()]
        else:
            encoder_module = None
            goal_rep_modules = []
        goal_rep_modules.extend(
            [
                (
                    SwiGLUBackbone(config["enc_hidden_dims"], config["feature_dim"])
                    if native
                    else MLP(
                        hidden_dims=(*config["enc_hidden_dims"], config["feature_dim"]),
                        activate_final=False,
                        layer_norm=config.get("layer_norm", True),
                    )
                ),
                LengthNormalize(),
            ]
        )
        goal_rep_def = nn.Sequential(goal_rep_modules)
        if encoder_module is not None:
            value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            high_actor_encoder_def = None
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config.get("layer_norm", True),
            ensemble=True,
            gc_encoder=value_encoder_def,
            backbone=SwiGLUBackbone(config["value_hidden_dims"], 1) if native else None,
        )
        target_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config.get("layer_norm", True),
            ensemble=True,
            gc_encoder=target_value_encoder_def,
            backbone=SwiGLUBackbone(config["value_hidden_dims"], 1) if native else None,
        )
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config["low_const_std"],
            gc_encoder=None,
            backbone=(
                SwiGLUBackbone(
                    config["actor_hidden_dims"], config["actor_hidden_dims"][-1]
                )
                if native
                else None
            ),
        )
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["feature_dim"],
            state_dependent_std=False,
            const_std=config["high_const_std"],
            gc_encoder=None,
            backbone=(
                SwiGLUBackbone(
                    config["actor_hidden_dims"], config["actor_hidden_dims"][-1]
                )
                if native
                else None
            ),
        )
        gc_encoder = (
            low_actor_encoder_def
            if config["gc_enc"] == "concat"
            else high_actor_encoder_def
        )
        actor_mixer_def = NavigationPolicy(
            state_dim=config["feature_dim"],
            high_actor_head=high_actor_def,
            low_actor_head=low_actor_def,
            num_mixer_blocks=config.get("num_mixer_blocks", 1),
            mixer_token_hidden=config.get("mixer_hidden", 32),
            mixer_channel_hidden=config.get("mixer_hidden", 32),
            gc_encoder=gc_encoder,
            enc_hidden=config["enc_hidden_dims"],
            feature_backbone=(
                SwiGLUBackbone(config["enc_hidden_dims"], config["feature_dim"])
                if native
                else None
            ),
            num_subgoals=config["num_subgoals"],
            causal_mixer=config.get("causal_mixer", True),
            planner_core_type=config.get("planner_core_type", "vanilla_coghp"),
            planner_block_type=config.get("planner_block_type", "swiglu_rmsnorm"),
            planner_num_layers=config.get("planner_num_layers", 1),
            planner_mixer_intermediate_size=config.get(
                "planner_mixer_intermediate_size", 256
            ),
            planner_h_cycles=config.get("planner_h_cycles", 2),
            planner_l_cycles=config.get("planner_l_cycles", 2),
            planner_bptt=config.get("planner_bptt", True),
            planner_norm_eps=config.get("planner_norm_eps", 1e-06),
            planner_forward_dtype=config.get("planner_forward_dtype", "float32"),
            share_actor_core=config.get("share_actor_core", False),
            action_hrm_num_layers=config["action_hrm_num_layers"],
            action_hrm_intermediate_size=config.get(
                "action_hrm_intermediate_size", 512
            ),
            action_hrm_head_dim=config.get("action_hrm_head_dim", 32),
            action_hrm_h_cycles=config["action_hrm_h_cycles"],
            action_hrm_l_cycles=config["action_hrm_l_cycles"],
            action_hrm_bptt=config["action_hrm_bptt"],
            action_hrm_is_causal=config["action_hrm_is_causal"],
            action_hrm_is_mlp_mixer=config.get("action_hrm_is_mlp_mixer", True),
            action_hrm_mlp_mixer_intermediate_size=config.get(
                "action_hrm_mlp_mixer_intermediate_size", 256
            ),
            action_core_type=config.get("action_core_type", "hrm"),
            action_block_type=config.get("action_block_type", "auto"),
            hrm_block_type=config.get("hrm_block_type", "legacy"),
            hrm_token_hidden=config.get("hrm_token_hidden", 32),
            hrm_channel_hidden=config.get("hrm_channel_hidden", 32),
            hrm_mixer_intermediate_size=config.get("hrm_mixer_intermediate_size", -1),
            hrm_layernorm_affine=config.get("hrm_layernorm_affine", True),
            action_hrm_norm_eps=config["action_hrm_norm_eps"],
            action_hrm_rope_theta=config.get("action_hrm_rope_theta", 10000.0),
            action_hrm_forward_dtype=config["action_hrm_forward_dtype"],
            train_action_chain_source=config["train_action_chain_source"],
        )
        network_info = {
            "goal_rep": (
                goal_rep_def,
                jnp.concatenate([ex_observations, ex_goals], axis=-1),
            ),
            "value": (value_def, (ex_observations, ex_goals)),
            "target_value": (target_value_def, (ex_observations, ex_goals)),
            "actor_mixer": (actor_mixer_def, (ex_observations, ex_goals, rng)),
        }
        network_def = ModuleDict(
            {key: value[0] for (key, value) in network_info.items()}
        )
        network_args = {key: value[1] for (key, value) in network_info.items()}
        network_variables = network_def.init(init_rng, **network_args)
        network_params = network_variables["params"]
        expected_action_core_params = config.get("expected_action_core_params", -1)
        if expected_action_core_params >= 0:
            actor_mixer_params = network_params["modules_actor_mixer"]
            action_core_type = config.get("action_core_type", "hrm")
            share_actor_core = config.get("share_actor_core", False)
            if share_actor_core:
                if "shared_actor_core" not in actor_mixer_params:
                    raise ValueError(
                        "Shared actor parameter contract was requested, but shared_actor_core was not created."
                    )
                action_core_params = actor_mixer_params["shared_actor_core"]
            elif action_core_type == "vanilla_coghp":
                action_mixer_params = {
                    key: value
                    for (key, value) in actor_mixer_params.items()
                    if key.startswith("action_mixer_blocks_")
                }
                if not action_mixer_params:
                    raise ValueError(
                        "Vanilla CoGHP action parameter contract was requested, but action_mixer_blocks_* were not created."
                    )
                action_core_params = action_mixer_params
            else:
                action_core_params = actor_mixer_params["action_hrm"]
            actual_action_core_params = sum(
                (
                    int(leaf.size)
                    for leaf in jax.tree_util.tree_leaves(action_core_params)
                )
            )
            if actual_action_core_params != expected_action_core_params:
                raise ValueError(
                    f"Action-core parameter contract failed: expected {expected_action_core_params}, got {actual_action_core_params}."
                )
        expected_planner_core_params = config.get("expected_planner_core_params", -1)
        if expected_planner_core_params >= 0:
            actor_mixer_params = network_params["modules_actor_mixer"]
            if config.get("share_actor_core", False):
                if "shared_actor_core" not in actor_mixer_params:
                    raise ValueError(
                        "Shared planner parameter contract was requested, but shared_actor_core was not created."
                    )
                planner_core_params = actor_mixer_params["shared_actor_core"]
            else:
                if "planner_core" not in actor_mixer_params:
                    raise ValueError(
                        "Planner parameter contract was requested, but planner_core was not created. Set planner_core_type=matched_feedforward or hrm."
                    )
                planner_core_params = actor_mixer_params["planner_core"]
            actual_planner_core_params = sum(
                (
                    int(leaf.size)
                    for leaf in jax.tree_util.tree_leaves(planner_core_params)
                )
            )
            if actual_planner_core_params != expected_planner_core_params:
                raise ValueError(
                    f"Planner-core parameter contract failed: expected {expected_planner_core_params}, got {actual_planner_core_params}."
                )
        network_params["modules_target_value"] = network_params["modules_value"]
        model_state = {
            collection: variables
            for (collection, variables) in network_variables.items()
            if collection != "params"
        }
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config["lr"]),
            model_state=model_state,
        )
        print("Creating navigation agent")
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    @property
    def position_mode(self):
        return self.config.get("command_content", "full_state") == "position"

    @property
    def global_mode(self):
        return self.config.get("high_weight_source", "detail_value") == "global_ota"

    def position_encode(self, observations, positions, params=None):
        return self.network.select("position_encoder")(
            observations, positions, params=params
        )

    def command_labels(self, batch, params=None):
        observations = jnp.repeat(
            batch["observations"][:, None, :], self.config["num_subgoals"], axis=1
        )
        if self.position_mode:
            return self.position_encode(
                observations, batch["high_actor_targets"][..., :2], params
            )
        return self.network.select("goal_rep")(
            jnp.concatenate([observations, batch["high_actor_targets"]], axis=-1),
            params=params,
        )

    def policy(
        self,
        observations,
        goals,
        rng,
        commands=None,
        params=None,
        temperature=1.0,
        chain_source=None,
        return_latents=False,
        return_diagnostics=False,
    ):
        kwargs = {}
        if self.position_mode:
            kwargs = dict(
                goal_rep_override=self.position_encode(
                    observations, goals[..., :2], params
                ),
                detach_planner_commands=True,
            )
        return self.network.select("actor_mixer")(
            observations,
            goals,
            rng,
            subgoal_reps=commands,
            params=params,
            temperature=temperature,
            action_chain_source=chain_source,
            return_latents=return_latents,
            return_diagnostics=return_diagnostics,
            **kwargs,
        )

    def high_advantages(self, batch):
        if self.global_mode:
            value = self.network.select("global_value")
            goals = batch["high_actor_goals"][..., :2]
        else:
            value = self.network.select("value")
            goals = batch["high_actor_goals"]
        v = jnp.mean(jnp.asarray(value(batch["observations"], goals)), axis=0)
        return jnp.stack(
            [
                jnp.mean(
                    jnp.asarray(value(batch["high_actor_targets"][:, j], goals)), axis=0
                )
                - v
                for j in range(self.config["num_subgoals"])
            ],
            axis=1,
        )

    def multi_high_actor_loss(self, batch, dist_list, obs_expand, grad_params):
        if not self.position_mode and (not self.global_mode):
            return self._detail_high_actor_loss(
                batch, dist_list, obs_expand, grad_params
            )
        targets = jax.lax.stop_gradient(self.command_labels(batch))
        advantages = self.high_advantages(batch)
        weights = stable_awr(
            advantages, self.config["high_alpha"], self.config["awr_max_weight"]
        )
        loss, mse, logp, std = (0.0, 0.0, 0.0, 0.0)
        info = {}
        k = self.config["num_subgoals"]
        for j in range(k):
            lp = dist_list[j].log_prob(targets[:, j])
            loss += (
                -(weights[:, j] * lp).mean()
                / self.config["subgoal_steps"]
                * self.config["high_discount"] ** (k - j - 1)
            )
            mse += jnp.mean((dist_list[j].mode() - targets[:, j]) ** 2)
            logp += lp.mean()
            std += dist_list[j].scale_diag.mean()
            info[f"adv_{j}/mean"] = advantages[:, j].mean()
            info[f"adv_{j}/std"] = advantages[:, j].std()
            info[f"weight_{j}/cap_fraction"] = jnp.mean(
                weights[:, j] >= self.config["awr_max_weight"] * (1 - 1e-06)
            )
            info[f"weight_{j}/mean"] = weights[:, j].mean()
            info[f"weight_{j}/ess_fraction"] = weights[:, j].sum() ** 2 / (
                weights.shape[0] * jnp.maximum(jnp.sum(weights[:, j] ** 2), 1e-12)
            )
        info.update(
            actor_loss=loss,
            adv=advantages.mean(),
            mse=mse / k,
            bc_log_prob=logp / k,
            std=std / k,
        )
        return (loss / k, info)

    def actor_loss(self, batch, grad_params, rng):
        if not self.position_mode:
            return self._full_state_actor_loss(batch, grad_params, rng)
        commands = self.command_labels(batch, grad_params)
        high, low, _, diagnostics = self.policy(
            batch["observations"],
            batch["high_actor_goals"],
            rng,
            commands,
            grad_params,
            return_diagnostics=True,
        )
        high_loss, high_info = self.multi_high_actor_loss(
            batch, high, None, grad_params
        )
        low_loss, low_info = self.low_actor_loss(batch, low)
        return (high_loss + low_loss, high_info, low_info, diagnostics)

    def position_loss(self, batch, params):
        positions = jnp.concatenate(
            [
                batch["high_actor_goals"][:, None, :2],
                batch["high_actor_targets"][..., :2],
            ],
            axis=1,
        )
        states = jnp.broadcast_to(
            batch["observations"][:, None, :],
            (*positions.shape[:-1], batch["observations"].shape[-1]),
        )
        latent = self.position_encode(states, positions, params)
        decoded = self.network.select("position_decoder")(latent, params=params)
        relative = (positions - states[..., :2]) / jnp.asarray(
            self.config["position_scale"]
        )
        loss = jnp.mean((decoded - relative) ** 2)
        return (
            loss,
            dict(
                reconstruction_mse=loss,
                latent_norm=jnp.linalg.norm(latent, axis=-1).mean(),
                conditional_spread=jnp.mean(jnp.var(latent, axis=1)),
            ),
        )

    def global_loss(self, batch, params):
        goals = batch["global_goals"]
        current_targets = self.network.select("target_global_value")(
            batch["observations"], goals
        )
        next_targets = self.network.select("target_global_value")(
            batch["global_observations"], goals
        )
        values = self.network.select("global_value")(
            batch["observations"], goals, params=params
        )
        loss = global_ivl_loss(
            values,
            current_targets,
            next_targets,
            batch["global_rewards"],
            batch["global_masks"],
            batch["global_valid"],
            self.config["global_discount"],
            self.config["expectile"],
        )
        valid = batch["global_valid"]
        den = jnp.maximum(valid.sum(), 1)
        return (
            loss,
            dict(
                loss=loss,
                v_mean=values.mean(),
                v_min=values.min(),
                v_max=values.max(),
                excluded_fraction=1 - valid.mean(),
                success_fraction=((1 - batch["global_masks"]) * valid).sum() / den,
                mean_duration=(batch["global_offsets"] * valid).sum() / den,
            ),
        )

    def loss_components(self, batch, params, rng):
        value, vi = self.value_loss(batch, params)
        actor, hi, li, di = self.actor_loss(batch, params, rng)
        high = hi["actor_loss"] / self.config["num_subgoals"]
        low = li["actor_loss"]
        info = {
            **{f"value/{k}": v for (k, v) in vi.items()},
            **{f"high_actor/{k}": v for (k, v) in hi.items()},
            **{f"low_actor/{k}": v for (k, v) in li.items()},
            **di,
        }
        pos, glob = (jnp.asarray(0.0), jnp.asarray(0.0))
        if self.position_mode:
            pos, pi = self.position_loss(batch, params)
            info.update({f"position/{k}": v for (k, v) in pi.items()})
        if self.global_mode:
            glob, gi = self.global_loss(batch, params)
            info.update({f"global/{k}": v for (k, v) in gi.items()})
        return (
            dict(
                value=value,
                high=high,
                low=low,
                position=pos,
                global_value=glob,
                actor=actor,
            ),
            info,
        )

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        if not self.position_mode and (not self.global_mode):
            return self._actor_value_loss(batch, grad_params, rng)
        losses, info = self.loss_components(
            batch, grad_params, self.rng if rng is None else rng
        )
        return (
            losses["value"]
            + losses["actor"]
            + losses["global_value"]
            + self.config["position_loss_coef"] * losses["position"],
            info,
        )

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        network, info = self.network.apply_loss_fn(
            lambda p: self.total_loss(batch, p, rng)
        )
        self.target_update(network, "value")
        if self.global_mode:
            self.target_update(network, "global_value")
        return (self.replace(network=network, rng=new_rng), info)

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        if not self.position_mode:
            return self._full_state_actions(observations, goals, seed, temperature)
        _, _, actions = self.policy(
            observations[None], goals[None], seed, temperature=temperature
        )
        return jnp.clip(actions[0], -1, 1)

    @jax.jit
    def sample_with_commands(self, observations, goals, positions, seed):
        commands = self.position_encode(
            jnp.broadcast_to(observations, (len(positions), len(observations))),
            positions,
        )
        _, dist, _ = self.policy(
            observations[None],
            goals[None],
            seed,
            commands[None],
            temperature=0.0,
            chain_source="teacher",
        )
        return jnp.clip(dist.mode()[0], -1, 1)

    @jax.jit
    def predicted_positions(self, observations, goals, seed):
        _, _, _, predicted = self.policy(
            observations[None], goals[None], seed, temperature=0.0, return_latents=True
        )
        offsets = self.network.select("position_decoder")(predicted[0])
        return observations[:2] + jnp.asarray(self.config["position_scale"]) * offsets

    @jax.jit
    def position_probe(self, batch):
        return self.command_labels(batch)

    @jax.jit
    def validation_rollout_info(self, batch, rng=None):
        if not self.position_mode:
            return self._full_state_validation_info(batch, rng)
        rng = self.rng if rng is None else rng
        commands = self.command_labels(batch)
        _, teacher, _ = self.policy(
            batch["observations"],
            batch["high_actor_goals"],
            rng,
            commands,
            temperature=0.0,
            chain_source="teacher",
        )
        _, predicted, _, latent = self.policy(
            batch["observations"],
            batch["high_actor_goals"],
            rng,
            temperature=0.0,
            return_latents=True,
        )
        tmse = jnp.mean((teacher.mode() - batch["actions"]) ** 2)
        pmse = jnp.mean((predicted.mode() - batch["actions"]) ** 2)
        scale = jnp.asarray(self.config["position_scale"])
        states = jnp.broadcast_to(
            batch["observations"][:, None],
            (*commands.shape[:-1], batch["observations"].shape[-1]),
        )
        target = batch["high_actor_targets"][..., :2]
        out = {
            "validation_action/teacher/mse": tmse,
            "validation_action/predicted/mse": pmse,
            "validation_action/gap/mse": pmse - tmse,
            "validation_position/predicted/latent_mse": jnp.mean(
                (latent - commands) ** 2
            ),
        }
        for name, z in [("teacher", commands), ("predicted", latent)]:
            q = states[..., :2] + scale * self.network.select("position_decoder")(z)
            recoded = self.position_encode(states, q)
            out[f"validation_position/{name}/position_error"] = jnp.linalg.norm(
                q - target, axis=-1
            ).mean()
            out[f"validation_position/{name}/cycle_mse"] = jnp.mean((recoded - z) ** 2)
            out[f"validation_position/{name}/norm"] = jnp.linalg.norm(z, axis=-1).mean()
        return out

    @jax.jit
    def gradient_diagnostics(self, batch):
        out = {}
        if self.position_mode:
            for loss_name in ("high", "low", "position"):
                grads = jax.grad(
                    lambda p: self.loss_components(batch, p, self.rng)[0][loss_name]
                )(self.network.params)
                for group in ("position_encoder", "position_decoder"):
                    leaves = jax.tree_util.tree_leaves(grads[f"modules_{group}"])
                    out[f"gradient/{loss_name}/{group}"] = jnp.sqrt(
                        sum((jnp.sum(x**2) for x in leaves))
                    )
        return out

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        if "command_content" not in config:
            return cls._create_core(seed, ex_observations, ex_actions, config)
        if config["command_content"] not in ("full_state", "position") or config[
            "high_weight_source"
        ] not in ("legacy_value", "detail_value", "global_ota"):
            raise ValueError("Unknown navigation condition.")
        if config["encoder"] is not None or config["frame_stack"] is not None:
            raise ValueError(
                "This study supports unstacked state-based navigation only."
            )
        base = cls._create_core(seed, ex_observations, ex_actions, config)
        position = config["command_content"] == "position"
        glob = config["high_weight_source"] == "global_ota"
        if not position and (not glob):
            return cls(base.rng, base.network, base.config)
        scale = tuple(config["position_scale"])
        mean = tuple(config["position_mean"])
        if len(scale) != 2 or not np.isfinite(scale).all() or min(scale) <= 0:
            raise ValueError("Finite positive training position scales required.")
        modules = dict(base.network.model_def.modules)
        args = dict(
            goal_rep=jnp.concatenate([ex_observations, ex_observations], axis=-1),
            value=(ex_observations, ex_observations),
            target_value=(ex_observations, ex_observations),
            actor_mixer=(ex_observations, ex_observations, base.rng),
        )
        if position:
            modules["position_encoder"] = PositionCommand(
                config["feature_dim"],
                scale,
                tuple(config["position_hidden_dims"]),
                network_backbone=config.get("network_backbone", "mlp"),
            )
            modules["position_decoder"] = PositionDecoder(
                tuple(config["position_hidden_dims"]),
                network_backbone=config.get("network_backbone", "mlp"),
            )
            args["position_encoder"] = (ex_observations, ex_observations[..., :2])
            args["position_decoder"] = jnp.ones(
                (*ex_observations.shape[:-1], config["feature_dim"])
            )
        if glob:
            for name in ("global_value", "target_global_value"):
                modules[name] = GlobalPotential(
                    config["global_dim"],
                    mean,
                    scale,
                    tuple(config["global_hidden_dims"]),
                    network_backbone=config.get("network_backbone", "mlp"),
                )
                args[name] = (ex_observations, ex_observations[..., :2])
        definition = ModuleDict(modules)
        variables = definition.init(
            jax.random.fold_in(jax.random.PRNGKey(seed), 9127), **args
        )
        params = variables["params"]
        params.update(copy.deepcopy(base.network.params))
        if glob:
            params["modules_target_global_value"] = copy.deepcopy(
                params["modules_global_value"]
            )
        state = {k: v for (k, v) in variables.items() if k != "params"}
        for collection, value in flax.core.unfreeze(base.network.model_state).items():
            state.setdefault(collection, {}).update(value)
        network = TrainState.create(
            definition, params, tx=optax.adam(config["lr"]), model_state=state
        )
        return cls(base.rng, network, flax.core.FrozenDict(config))


def _legacy_base_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="coghp",
            lr=0.0003,
            batch_size=256,
            discount=0.99,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            alpha=1.0,
            tau=0.005,
            expectile=0.7,
            low_alpha=3.0,
            high_alpha=3.0,
            subgoal_steps=25,
            low_actor_rep_grad=False,
            high_const_std=True,
            low_const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class="MultiHGCDataset",
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
            feature_dim=32,
            mixer_hidden=32,
            num_mixer_blocks=1,
            enc_hidden_dims=(512, 512, 512),
            layer_norm=True,
            action_chunk=ml_collections.config_dict.placeholder(int),
            num_subgoals=1,
            causal_mixer=True,
            action_use_full_subgoal_chain=True,
            share_mixer_weights=False,
            separate_action_mixer=False,
            gc_enc="concat",
            high_discount=0.8,
        )
    )
    return config


def get_legacy_action_config():
    config = _legacy_base_config()
    config.agent_name = "hrm_action_coghp"
    config.planner_core_type = "vanilla_coghp"
    config.planner_block_type = "swiglu_rmsnorm"
    config.planner_num_layers = 1
    config.planner_mixer_intermediate_size = 256
    config.planner_h_cycles = 2
    config.planner_l_cycles = 2
    config.planner_bptt = True
    config.planner_norm_eps = 1e-06
    config.planner_forward_dtype = "float32"
    config.share_actor_core = False
    config.expected_planner_core_params = -1
    config.action_hrm_num_layers = 1
    config.action_hrm_intermediate_size = 512
    config.action_hrm_head_dim = 32
    config.action_hrm_h_cycles = 2
    config.action_hrm_l_cycles = 2
    config.action_hrm_bptt = True
    config.action_hrm_is_causal = True
    config.action_hrm_is_mlp_mixer = True
    config.action_hrm_mlp_mixer_intermediate_size = 256
    config.action_core_type = "hrm"
    config.action_block_type = "auto"
    config.hrm_block_type = "legacy"
    config.hrm_token_hidden = 32
    config.hrm_channel_hidden = 32
    config.hrm_mixer_intermediate_size = -1
    config.hrm_layernorm_affine = True
    config.expected_action_core_params = -1
    config.action_hrm_norm_eps = 1e-06
    config.action_hrm_rope_theta = 10000.0
    config.action_hrm_forward_dtype = "float32"
    config.train_action_chain_source = "teacher"
    config.enable_action_hrm_validation = True
    return config


def get_legacy_navigation_config():
    config = get_legacy_action_config()
    config.agent_name = "navigation_coghp"
    config.command_content = "full_state"
    config.high_weight_source = "legacy_value"
    config.position_scale = (1.0, 1.0)
    config.position_mean = (0.0, 0.0)
    config.position_hidden_dims = (128, 128)
    config.position_loss_coef = 1.0
    config.global_dim = 128
    config.global_hidden_dims = (512, 512, 512)
    config.global_abstraction_steps = 25
    config.global_discount = 0.99
    config.global_goal_tolerance = -1.0
    config.awr_max_weight = 100.0
    config.navigation_protocol = "position_ota_v1"
    return config


def get_config(profile=None):
    """Native method defaults; historical profiles are explicit compatibility inputs."""
    if profile == "legacy_action":
        return get_legacy_action_config()
    if profile == "legacy_navigation":
        return get_legacy_navigation_config()
    if profile not in (None, "", "default"):
        raise ValueError(f"Unknown navigation profile: {profile!r}")
    return ml_collections.ConfigDict(
        {
            "action_block_type": "swiglu_rmsnorm",
            "action_core_type": "hrm",
            "action_hrm_bptt": True,
            "action_hrm_forward_dtype": "float32",
            "action_hrm_h_cycles": 2,
            "action_hrm_is_causal": True,
            "action_hrm_l_cycles": 2,
            "action_hrm_norm_eps": 1e-06,
            "action_hrm_num_layers": 1,
            "actor_geom_sample": False,
            "actor_hidden_dims": (512, 512, 512),
            "actor_p_curgoal": 0.0,
            "actor_p_randomgoal": 0.0,
            "actor_p_trajgoal": 1.0,
            "agent_name": "navigation",
            "awr_max_weight": 100.0,
            "batch_size": 256,
            "causal_mixer": True,
            "command_content": "full_state",
            "dataset_class": "MultiHGCDataset",
            "discount": 0.99,
            "discrete": False,
            "enable_action_hrm_validation": True,
            "enc_hidden_dims": (512, 512, 512),
            "encoder": ml_collections.config_dict.placeholder(str),
            "expected_action_core_params": -1,
            "expected_planner_core_params": -1,
            "expectile": 0.7,
            "feature_dim": 32,
            "frame_stack": ml_collections.config_dict.placeholder(int),
            "gc_enc": "concat",
            "gc_negative": True,
            "global_abstraction_steps": 25,
            "global_dim": 128,
            "global_discount": 0.99,
            "global_goal_tolerance": -1.0,
            "global_hidden_dims": (512, 512, 512),
            "high_alpha": 3.0,
            "high_const_std": True,
            "high_discount": 0.8,
            "high_weight_source": "detail_value",
            "hrm_mixer_intermediate_size": 256,
            "low_alpha": 3.0,
            "low_const_std": True,
            "lr": 0.0003,
            "navigation_protocol": "navigation_v1",
            "network_backbone": "swiglu_rmsnorm",
            "num_subgoals": 1,
            "p_aug": 0.0,
            "planner_block_type": "swiglu_rmsnorm",
            "planner_bptt": True,
            "planner_core_type": "matched_feedforward",
            "planner_forward_dtype": "float32",
            "planner_h_cycles": 2,
            "planner_l_cycles": 2,
            "planner_mixer_intermediate_size": 256,
            "planner_norm_eps": 1e-06,
            "planner_num_layers": 1,
            "position_hidden_dims": (128, 128),
            "position_loss_coef": 1.0,
            "position_mean": (0.0, 0.0),
            "position_scale": (1.0, 1.0),
            "share_actor_core": False,
            "subgoal_steps": 25,
            "tau": 0.005,
            "train_action_chain_source": "teacher",
            "value_geom_sample": True,
            "value_hidden_dims": (512, 512, 512),
            "value_p_curgoal": 0.2,
            "value_p_randomgoal": 0.3,
            "value_p_trajgoal": 0.5,
        }
    )
