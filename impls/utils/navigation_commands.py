"""Independent position commands and bilinear global values for navigation."""

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class PositionCommand(nn.Module):
    dim: int
    scale: tuple
    hidden_dims: tuple = (128, 128)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, states, positions):
        scale = self.variable('constants', 'position_scale',
                              lambda: jnp.asarray(self.scale, dtype=jnp.float32)).value
        relative = (positions - states[..., :2]) / scale
        latent = MLP((*self.hidden_dims, self.dim), layer_norm=True,
                     name='encoder')(jnp.concatenate([states, relative], axis=-1))
        # max BEFORE sqrt also makes the derivative finite at exactly zero.
        norm = jnp.sqrt(jnp.maximum(jnp.sum(latent ** 2, axis=-1, keepdims=True), self.eps ** 2))
        return latent * jnp.sqrt(float(self.dim)) / norm


class PositionDecoder(nn.Module):
    hidden_dims: tuple = (128, 128)

    @nn.compact
    def __call__(self, latent):
        return MLP((*self.hidden_dims, 2), layer_norm=True, name='decoder')(latent)


class GlobalPotential(nn.Module):
    dim: int
    position_mean: tuple
    position_scale: tuple
    hidden_dims: tuple = (512, 512, 512)

    @nn.compact
    def __call__(self, states, positions):
        mean = self.variable('constants', 'position_mean',
                             lambda: jnp.asarray(self.position_mean, dtype=jnp.float32)).value
        scale = self.variable('constants', 'position_scale',
                              lambda: jnp.asarray(self.position_scale, dtype=jnp.float32)).value
        goals = (positions - mean) / scale
        values = []
        for i in range(2):
            u = MLP((*self.hidden_dims, self.dim), layer_norm=True,
                    name=f'state_{i}')(states)
            c = MLP((*self.hidden_dims, self.dim), layer_norm=True,
                    name=f'goal_{i}')(goals)
            values.append(jnp.sum(u * c, axis=-1) / jnp.sqrt(float(self.dim)))
        return jnp.stack(values, axis=0)
