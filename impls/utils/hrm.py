"""Faithful Flax port of the standalone HRM-Mini architecture.

This module mirrors ``hrm-mini/arch/layers.py`` and
``hrm-mini/arch/hrm.py`` without adding CoGHP-specific concepts.  In
particular, it deliberately contains no subgoal stages, action residuals,
stage embeddings, actor heads, or context-vector compression.

The port preserves the architectural details that matter for an HRM core:

* SwiGLU token/channel mixing and bias-free linear layers.
* Post-RMSNorm after each residual branch.
* Sequence-shaped ``z_H`` and ``z_L`` recurrent states.
* Low-first, high-last H/L update scheduling.
* Output from the final high-level state.
* Full inner BPTT or the official one-step-gradient ablation.
* Detached carry between outer recurrent calls.
* Non-trainable initial states stored in a Flax ``constants`` collection.

The original tuned HRM uses attention by default and exposes MLP-Mixer as an
ablation.  Both backbones are implemented here so later CoGHP experiments can
separate the HRM recurrence from the choice of sequence mixer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


Array = jax.Array
CosSin = Tuple[Array, Array]
Carry = Mapping[str, Array]


@dataclass(frozen=True, kw_only=True)
class TransformerConfig:
    """Configuration fields from HRM-Mini's ``TransformerConfig``."""

    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    norm_eps: float
    rope_theta: float
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256


@dataclass(frozen=True, kw_only=True)
class HRMConfig(TransformerConfig):
    """Configuration fields from HRM-Mini's ``HRMConfig``."""

    vocab_size: int
    H_cycles: int
    L_cycles: int
    bptt: bool
    forward_dtype: str


def approximate_truncated_normal(std: float = 1.0):
    """Return the approximate truncated-normal initializer used by HRM-Mini.

    HRM-Mini samples a standard normal, applies ``fmod(3)``, and multiplies by
    ``1.014762601732121``.  This intentionally mirrors that implementation
    instead of substituting JAX's statistically exact truncated normal.
    """

    def init(key, shape, dtype=jnp.float32):
        values = jax.random.normal(key, shape, dtype=dtype)
        return (
            jnp.fmod(values, jnp.asarray(3.0, dtype=dtype))
            * jnp.asarray(1.014762601732121 * std, dtype=dtype)
        )

    return init


def _fan_in_truncated_normal(key, shape, dtype=jnp.float32):
    if len(shape) < 2:
        raise ValueError(f'Linear kernel must have rank at least 2, got {shape}.')
    fan_in = shape[-2]
    return approximate_truncated_normal(1.0 / math.sqrt(fan_in))(
        key,
        shape,
        dtype,
    )


def rms_norm(x: Array, epsilon: float) -> Array:
    """Parameter-free RMSNorm equivalent to ``torch.nn.functional.rms_norm``."""

    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    normalized = x_float * jax.lax.rsqrt(variance + epsilon)
    return normalized.astype(x.dtype)


def rotate_half(x: Array) -> Array:
    """Rotate the two halves of the final dimension for RoPE."""

    if x.shape[-1] % 2 != 0:
        raise ValueError(f'RoPE head dimension must be even, got {x.shape[-1]}.')
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-second, first], axis=-1)


def apply_rotary_pos_emb(x: Array, cos_sin: CosSin) -> Array:
    """Apply HRM-Mini's rotary positional embedding permutation."""

    cos, sin = cos_sin
    broadcast_shape = (1,) * (x.ndim - 3) + (cos.shape[0], 1, cos.shape[1])
    cos = jnp.reshape(cos, broadcast_shape)
    sin = jnp.reshape(sin, broadcast_shape)
    return (x * cos + rotate_half(x) * sin).astype(x.dtype)


class CastedLinear(nn.Module):
    """Bias-optional linear layer with HRM-Mini initialization and casting."""

    in_features: int
    out_features: int
    bias: bool
    batch_output_dims: Tuple[int, ...] = ()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f'Expected input width {self.in_features}, got {x.shape[-1]}.'
            )
        flattened_out_features = math.prod(self.batch_output_dims) * self.out_features
        return nn.Dense(
            features=flattened_out_features,
            use_bias=self.bias,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_fan_in_truncated_normal,
            bias_init=nn.initializers.zeros,
            name='linear',
        )(x)


class CastedScaledEmbedding(nn.Module):
    """Embedding whose output is scaled to unit standard deviation."""

    num_embeddings: int
    embedding_dim: int
    forward_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, input_ids: Array) -> Array:
        scale = math.sqrt(self.embedding_dim)
        embedding = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            dtype=self.forward_dtype,
            param_dtype=self.param_dtype,
            embedding_init=approximate_truncated_normal(1.0 / scale),
            name='embedding',
        )(input_ids)
        return (scale * embedding).astype(self.forward_dtype)


class MLP(nn.Module):
    """Bias-free GELU MLP used by the attention HRM backbone."""

    hidden_size: int
    intermediate_size: int
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        hidden = CastedLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            param_dtype=self.param_dtype,
            name='up_proj',
        )(x)
        hidden = jax.nn.gelu(hidden, approximate=False)
        return CastedLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            name='down_proj',
        )(hidden)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU matching HRM-Mini's mixer feed-forward modules."""

    hidden_size: int
    intermediate_size: int
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        gate_up = CastedLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            batch_output_dims=(2,),
            param_dtype=self.param_dtype,
            name='gate_up_proj',
        )(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = jax.nn.silu(gate) * up
        return CastedLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            name='down_proj',
        )(hidden)


class RotaryEmbedding(nn.Module):
    """Stateless rotary embedding table used by the attention backbone."""

    dim: int
    max_position_embeddings: int
    base: float = 10000.0

    @nn.compact
    def __call__(self) -> CosSin:
        if self.dim % 2 != 0:
            raise ValueError(f'RoPE dimension must be even, got {self.dim}.')
        indices = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (self.base ** (indices / self.dim))
        positions = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        frequencies = jnp.outer(positions, inv_freq)
        embeddings = jnp.concatenate([frequencies, frequencies], axis=-1)
        return jnp.cos(embeddings), jnp.sin(embeddings)


class Attention(nn.Module):
    """Multi-head attention matching HRM-Mini's projection and RoPE layout."""

    hidden_size: int
    head_dim: int
    num_heads: int
    is_causal: bool
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, cos_sin: CosSin) -> Array:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                'hidden_size must equal num_heads * head_dim; got '
                f'{self.hidden_size}, {self.num_heads}, {self.head_dim}.'
            )

        # HRM-Mini flattens a (3, hidden_size, input_size) projection and then
        # reshapes the final axis as (num_heads, 3 * head_dim) before chunking.
        qkv = CastedLinear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            batch_output_dims=(3,),
            param_dtype=self.param_dtype,
            name='qkv_proj',
        )(hidden_states)
        qkv = jnp.reshape(
            qkv,
            (*qkv.shape[:-1], self.num_heads, 3 * self.head_dim),
        )
        query, key, value = jnp.split(qkv, 3, axis=-1)
        query = apply_rotary_pos_emb(query, cos_sin)
        key = apply_rotary_pos_emb(key, cos_sin)

        mask = None
        if self.is_causal:
            mask = nn.make_causal_mask(
                jnp.ones(hidden_states.shape[:-1], dtype=jnp.bool_)
            )
        output = nn.dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            deterministic=True,
        )
        output = jnp.reshape(output, (*output.shape[:-2], self.hidden_size))
        return CastedLinear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            name='o_proj',
        )(output)


class TransformerBlock(nn.Module):
    """Official HRM post-norm attention or MLP-Mixer block."""

    seq_len: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    norm_eps: float = 1e-6
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array, cos_sin: Optional[CosSin] = None) -> Array:
        if x.shape[-2:] != (self.seq_len, self.hidden_size):
            raise ValueError(
                f'Expected [..., {self.seq_len}, {self.hidden_size}], got '
                f'{x.shape}.'
            )

        if self.is_mlp_mixer:
            token_input = jnp.swapaxes(x, -1, -2)
            token_update = SwiGLU(
                hidden_size=self.seq_len,
                intermediate_size=self.mlp_mixer_intermediate_size,
                param_dtype=self.param_dtype,
                name='mlp_t',
            )(token_input)
            x = jnp.swapaxes(
                rms_norm(token_input + token_update, self.norm_eps),
                -1,
                -2,
            )
        else:
            if cos_sin is None:
                raise ValueError('cos_sin is required by the attention block.')
            attention_update = Attention(
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                num_heads=self.hidden_size // self.head_dim,
                is_causal=self.is_causal,
                param_dtype=self.param_dtype,
                name='attn',
            )(x, cos_sin)
            x = rms_norm(x + attention_update, self.norm_eps)

        if self.is_mlp_mixer:
            channel_update = SwiGLU(
                hidden_size=self.hidden_size,
                intermediate_size=self.mlp_mixer_intermediate_size,
                param_dtype=self.param_dtype,
                name='mlp',
            )(x)
        else:
            channel_update = MLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                param_dtype=self.param_dtype,
                name='mlp',
            )(x)
        return rms_norm(x + channel_update, self.norm_eps)


class Transformer(nn.Module):
    """Stack of HRM-Mini TransformerBlocks."""

    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array) -> Array:
        cos_sin = None
        if not self.is_mlp_mixer:
            cos_sin = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=self.seq_len,
                base=self.rope_theta,
                name='rotary_emb',
            )()

        output = hidden_states
        for layer_idx in range(self.num_layers):
            output = TransformerBlock(
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                head_dim=self.head_dim,
                is_causal=self.is_causal,
                norm_eps=self.norm_eps,
                is_mlp_mixer=self.is_mlp_mixer,
                mlp_mixer_intermediate_size=self.mlp_mixer_intermediate_size,
                param_dtype=self.param_dtype,
                name=f'layers_{layer_idx}',
            )(output, cos_sin=cos_sin)
        return output


class HRMCore(nn.Module):
    """Continuous-input HRM core corresponding to HRM-Mini's recurrence."""

    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    h_cycles: int
    l_cycles: int
    bptt: bool = True
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    forward_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        if self.seq_len < 1 or self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError('seq_len, hidden_size, and num_layers must be positive.')
        if self.h_cycles < 1 or self.l_cycles < 1:
            raise ValueError('h_cycles and l_cycles must both be positive.')
        if not self.is_mlp_mixer:
            if self.head_dim < 1 or self.hidden_size % self.head_dim != 0:
                raise ValueError(
                    'Attention HRM requires head_dim to divide hidden_size.'
                )

        level_kwargs = dict(
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            is_causal=self.is_causal,
            norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
            is_mlp_mixer=self.is_mlp_mixer,
            mlp_mixer_intermediate_size=self.mlp_mixer_intermediate_size,
            param_dtype=self.param_dtype,
        )
        self.H_level = Transformer(**level_kwargs, name='H_level')
        self.L_level = Transformer(**level_kwargs, name='L_level')

        # PyTorch buffers are represented as a non-trainable Flax collection.
        # The vectors broadcast across batch and sequence dimensions, exactly
        # as in HRM-Mini.
        self.zH_init = self.variable(
            'constants',
            'zH_init',
            lambda: approximate_truncated_normal(1.0)(
                self.make_rng('params'),
                (self.hidden_size,),
                self.forward_dtype,
            ),
        )
        self.zL_init = self.variable(
            'constants',
            'zL_init',
            lambda: approximate_truncated_normal(1.0)(
                self.make_rng('params'),
                (self.hidden_size,),
                self.forward_dtype,
            ),
        )

    def initial_carry(self, batch_shape: Tuple[int, ...]) -> dict[str, Array]:
        """Broadcast the fixed HRM initial vectors to a sequence-shaped carry."""

        shape = (*batch_shape, self.seq_len, self.hidden_size)
        return {
            'z_H': jnp.broadcast_to(self.zH_init.value, shape),
            'z_L': jnp.broadcast_to(self.zL_init.value, shape),
        }

    def __call__(
        self,
        carry: Optional[Carry],
        inputs: Array,
    ) -> tuple[dict[str, Array], Array]:
        x = inputs.astype(self.forward_dtype)
        if x.shape[-2:] != (self.seq_len, self.hidden_size):
            raise ValueError(
                f'Expected [..., {self.seq_len}, {self.hidden_size}], got '
                f'{x.shape}.'
            )
        if carry is None:
            carry = self.initial_carry(x.shape[:-2])

        z_H = carry['z_H'].astype(self.forward_dtype)
        z_L = carry['z_L'].astype(self.forward_dtype)
        expected_shape = x.shape
        if z_H.shape != expected_shape or z_L.shape != expected_shape:
            raise ValueError(
                f'Carry and input shapes must match; got x={x.shape}, '
                f'z_H={z_H.shape}, z_L={z_L.shape}.'
            )

        # Match HRM-Mini exactly: run all but the final low update in the
        # optional BPTT region, update H after every completed L group, then
        # always execute one final differentiable L/H pair.
        for iteration in range(self.h_cycles * self.l_cycles - 1):
            z_L = self.L_level(z_L + z_H + x)
            if (iteration + 1) % self.l_cycles == 0:
                z_H = self.H_level(z_H + z_L)

        if not self.bptt:
            z_H = jax.lax.stop_gradient(z_H)
            z_L = jax.lax.stop_gradient(z_L)

        z_L = self.L_level(z_L + z_H + x)
        z_H = self.H_level(z_H + z_L)

        next_carry = {
            'z_H': jax.lax.stop_gradient(z_H),
            'z_L': jax.lax.stop_gradient(z_L),
        }
        return next_carry, z_H


class HRM(nn.Module):
    """Token-input HRM wrapper corresponding to ``hrm-mini/arch/hrm.py``."""

    vocab_size: int
    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    h_cycles: int
    l_cycles: int
    bptt: bool = True
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    forward_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.embed = CastedScaledEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            forward_dtype=self.forward_dtype,
            param_dtype=self.param_dtype,
            name='embed',
        )
        self.core = HRMCore(
            seq_len=self.seq_len,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            is_causal=self.is_causal,
            h_cycles=self.h_cycles,
            l_cycles=self.l_cycles,
            bptt=self.bptt,
            norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
            is_mlp_mixer=self.is_mlp_mixer,
            mlp_mixer_intermediate_size=self.mlp_mixer_intermediate_size,
            forward_dtype=self.forward_dtype,
            param_dtype=self.param_dtype,
            name='core',
        )
        self.lm_head = CastedLinear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            bias=False,
            param_dtype=self.param_dtype,
            name='lm_head',
        )

    def initial_carry(self, batch_shape: Tuple[int, ...]) -> dict[str, Array]:
        return self.core.initial_carry(batch_shape)

    def __call__(
        self,
        carry: Optional[Carry],
        input_ids: Array,
    ) -> tuple[dict[str, Array], Array]:
        if input_ids.shape[-1] != self.seq_len:
            raise ValueError(
                f'Expected input sequence length {self.seq_len}, got '
                f'{input_ids.shape[-1]}.'
            )
        embedded = self.embed(input_ids)
        next_carry, z_H = self.core(carry, embedded)
        return next_carry, self.lm_head(z_H)


__all__ = [
    'Attention',
    'Carry',
    'CastedLinear',
    'CastedScaledEmbedding',
    'CosSin',
    'HRM',
    'HRMConfig',
    'HRMCore',
    'MLP',
    'RotaryEmbedding',
    'SwiGLU',
    'Transformer',
    'TransformerBlock',
    'TransformerConfig',
    'apply_rotary_pos_emb',
    'approximate_truncated_normal',
    'rms_norm',
    'rotate_half',
]
