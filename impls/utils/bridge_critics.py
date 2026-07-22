"""Critics used to rerank FM-CoGHP subgoal-chain candidates."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import Identity, MLP


class LocalCRLBridgeCritic(nn.Module):
    """A local contrastive state-to-encoded-goal critic."""

    latent_dim: int
    temperature: float
    hidden_dims: Sequence[int] = (256, 256)
    state_encoder: nn.Module = None
    layer_norm: bool = True

    def setup(self):
        self.raw_state_encoder = self.state_encoder if self.state_encoder is not None else Identity()
        self.state_projection = MLP(
            hidden_dims=(*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.goal_projection = MLP(
            hidden_dims=(*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    @staticmethod
    def normalize(x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

    def encode_states(self, observations):
        return self.normalize(self.state_projection(self.raw_state_encoder(observations)))

    def encode_goals(self, goal_reps):
        return self.normalize(self.goal_projection(goal_reps))

    def __call__(self, observations, goal_reps, pairwise=False):
        state_embeddings = self.encode_states(observations)
        goal_embeddings = self.encode_goals(goal_reps)
        if pairwise:
            return (state_embeddings @ goal_embeddings.T) / self.temperature
        if goal_embeddings.ndim == state_embeddings.ndim + 1:
            return jnp.einsum('bd,bmd->bm', state_embeddings, goal_embeddings) / self.temperature
        return jnp.sum(state_embeddings * goal_embeddings, axis=-1) / self.temperature
