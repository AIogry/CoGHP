"""Flow-matching network components for FM-CoGHP."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.coghp_network import MixerBlock
from utils.networks import LengthNormalize, MLP, default_init


class FlowTimeEmbedding(nn.Module):
    """Fourier features followed by an MLP projection."""

    embed_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, flow_times):
        flow_times = jnp.asarray(flow_times, dtype=jnp.float32).reshape(-1, 1)
        half_dim = max((self.embed_dim + 1) // 2, 1)
        frequency_exponents = jnp.arange(half_dim, dtype=flow_times.dtype) / max(half_dim - 1, 1)
        frequencies = jnp.exp(-jnp.log(10000.0) * frequency_exponents)
        angles = 2.0 * jnp.pi * flow_times * frequencies[None, :]
        embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        embedding = embedding[:, : self.embed_dim]
        return MLP(
            hidden_dims=(self.output_dim, self.output_dim),
            activate_final=False,
            layer_norm=True,
        )(embedding)


class FlowChainPlanner(nn.Module):
    """Conditional vector field that jointly updates a latent subgoal chain."""

    num_subgoals: int
    feature_dim: int
    num_mixer_blocks: int
    mixer_token_hidden: int
    mixer_channel_hidden: int
    flow_time_embed_dim: int
    gc_encoder: nn.Module = None
    enc_hidden: Sequence[int] = (128, 128)
    layer_norm: bool = True

    def setup(self):
        self.time_embedding = FlowTimeEmbedding(
            embed_dim=self.flow_time_embed_dim,
            output_dim=self.feature_dim,
        )
        self.slot_embeddings = self.param(
            'slot_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_subgoals, self.feature_dim),
        )
        self.obs_embedding = nn.Sequential(
            [
                MLP(
                    hidden_dims=(*self.enc_hidden, self.feature_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm,
                ),
                LengthNormalize(),
            ]
        )
        self.mixer_blocks = [
            MixerBlock(
                num_tokens=self.num_subgoals + 2,
                embed_dim=self.feature_dim,
                hidden_dim_tokens=self.mixer_token_hidden,
                hidden_dim_channels=self.mixer_channel_hidden,
                causal=False,
            )
            for _ in range(self.num_mixer_blocks)
        ]
        self.velocity_head = nn.Dense(
            self.feature_dim,
            kernel_init=default_init(1e-2),
        )
        self.final_layer_norm = nn.LayerNorm() if self.layer_norm else None

    def __call__(self, observations, goals, chain_t, flow_times):
        observations = jnp.expand_dims(observations, axis=1)
        goals = jnp.expand_dims(goals, axis=1) if goals is not None else None

        if self.gc_encoder is not None:
            condition_features = self.gc_encoder(
                observations,
                goals,
                goal_encoded=False,
                listwise=True,
            )
            obs_feature = self.obs_embedding(condition_features[0])
            goal_feature = condition_features[1]
        else:
            obs_feature = self.obs_embedding(observations)
            goal_feature = self.obs_embedding(goals)

        time_feature = self.time_embedding(flow_times)
        chain_tokens = (
            chain_t
            + time_feature[:, None, :]
            + self.slot_embeddings[None, :, :]
        )
        tokens = jnp.concatenate([obs_feature, goal_feature, chain_tokens], axis=1)
        for mixer_block in self.mixer_blocks:
            tokens = mixer_block(tokens)
        if self.final_layer_norm is not None:
            tokens = self.final_layer_norm(tokens)

        return self.velocity_head(tokens[:, 2:, :])
