"""Navigation backbones and their local feed-forward/normalization helpers.

H/L scheduling follows HRM-Mini; this module has no token vocabulary, language
model wrapper, or external model configuration. GELU and attention variants
remain explicit controls for restoring historical experiments. New navigation
models use the SwiGLU/RMSNorm paths.
"""

from __future__ import annotations
import math
from typing import Any, Mapping, Optional, Tuple, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
CosSin = Tuple[Array, Array]
Carry = Mapping[str, Array]

MIXER_BLOCK_TYPES = frozenset({"swiglu_rmsnorm", "coghp_gelu", "gelu_layernorm"})

BACKBONE_TYPES = frozenset({"matched_feedforward", "single_state_iterative", "hrm"})

_ATTENTION_BLOCK_TYPE = "attention"


def _resolve_block_type(block_type: Optional[str], is_mlp_mixer: bool) -> str:
    """Resolve the new block selector while preserving the legacy bool API."""
    if block_type in (None, "", "auto", "legacy"):
        return "swiglu_rmsnorm" if is_mlp_mixer else _ATTENTION_BLOCK_TYPE
    allowed_block_types = MIXER_BLOCK_TYPES | {_ATTENTION_BLOCK_TYPE}
    if block_type not in allowed_block_types:
        raise ValueError(
            f"Unknown HRM block type {block_type!r}; expected one of {sorted(allowed_block_types)}."
        )
    return block_type


def approximate_truncated_normal(std: float = 1.0):
    """Return the approximate truncated-normal initializer used by HRM-Mini.

    HRM-Mini samples a standard normal, applies ``fmod(3)``, and multiplies by
    ``1.014762601732121``.  This intentionally mirrors that implementation
    instead of substituting JAX's statistically exact truncated normal.
    """

    def init(key, shape, dtype=jnp.float32):
        values = jax.random.normal(key, shape, dtype=dtype)
        return jnp.fmod(values, jnp.asarray(3.0, dtype=dtype)) * jnp.asarray(
            1.014762601732121 * std, dtype=dtype
        )

    return init


def _fan_in_truncated_normal(key, shape, dtype=jnp.float32):
    if len(shape) < 2:
        raise ValueError(f"Linear kernel must have rank at least 2, got {shape}.")
    fan_in = shape[-2]
    return approximate_truncated_normal(1.0 / math.sqrt(fan_in))(key, shape, dtype)


def rms_norm(x: Array, epsilon: float) -> Array:
    """Parameter-free RMSNorm equivalent to ``torch.nn.functional.rms_norm``."""
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    normalized = x_float * jax.lax.rsqrt(variance + epsilon)
    return normalized.astype(x.dtype)


def rotate_half(x: Array) -> Array:
    """Rotate the two halves of the final dimension for RoPE."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {x.shape[-1]}.")
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-second, first], axis=-1)


def apply_rotary_pos_emb(x: Array, cos_sin: CosSin) -> Array:
    """Apply HRM-Mini's rotary positional embedding permutation."""
    cos, sin = cos_sin
    broadcast_shape = (1,) * (x.ndim - 3) + (cos.shape[0], 1, cos.shape[1])
    cos = jnp.reshape(cos, broadcast_shape)
    sin = jnp.reshape(sin, broadcast_shape)
    return (x * cos + rotate_half(x) * sin).astype(x.dtype)


class Linear(nn.Module):
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
                f"Expected input width {self.in_features}, got {x.shape[-1]}."
            )
        flattened_out_features = math.prod(self.batch_output_dims) * self.out_features
        return nn.Dense(
            features=flattened_out_features,
            use_bias=self.bias,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_fan_in_truncated_normal,
            bias_init=nn.initializers.zeros,
            name="linear",
        )(x)


class AttentionFeedForward(nn.Module):
    """Bias-free GELU MLP used by the attention HRM backbone."""

    hidden_size: int
    intermediate_size: int
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        hidden = Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            param_dtype=self.param_dtype,
            name="up_proj",
        )(x)
        hidden = jax.nn.gelu(hidden, approximate=False)
        return Linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(hidden)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU matching HRM-Mini's mixer feed-forward modules."""

    hidden_size: int
    intermediate_size: int
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        gate_up = Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            batch_output_dims=(2,),
            param_dtype=self.param_dtype,
            name="gate_up_proj",
        )(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = jax.nn.silu(gate) * up
        return Linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(hidden)


def _comparison_init(scale: float = 1.0):
    """Independent copy of CoGHP's default dense-kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class GELUMixerBlock(nn.Module):
    """Exact local mirror of CoGHP's GELU MixerBlock.

    This implementation intentionally lives in ``hrm.py``.  It mirrors the
    Dense biases, initialization, additional ``tm_weights`` matrix, causal
    triangular projection, token/channel mixing, and residual placement of
    ``utils.coghp_network.MixerBlock`` without importing that class.
    """

    seq_len: int
    hidden_size: int
    token_hidden: int
    channel_hidden: int
    is_causal: bool = True
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if x.ndim != 3 or x.shape[-2:] != (self.seq_len, self.hidden_size):
            raise ValueError(
                f"Expected [B, {self.seq_len}, {self.hidden_size}], got {x.shape}."
            )
        token_update = jnp.swapaxes(x, -1, -2)
        token_update = nn.Dense(
            self.token_hidden,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="token_dense1",
        )(token_update)
        token_update = nn.gelu(token_update)
        token_update = nn.Dense(
            self.seq_len,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="token_dense2",
        )(token_update)
        token_update = jnp.swapaxes(token_update, -1, -2)
        tm_weights = self.param(
            "tm_weights",
            nn.initializers.normal(stddev=0.02),
            (self.seq_len, self.seq_len),
            self.param_dtype,
        )
        if self.is_causal:
            tm_weights = jnp.tril(tm_weights)
        token_update = jnp.einsum("btd,ts->bsd", token_update, tm_weights)
        x = x + token_update
        channel_update = nn.Dense(
            self.channel_hidden,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="channel_dense1",
        )(x)
        channel_update = nn.gelu(channel_update)
        channel_update = nn.Dense(
            self.hidden_size,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="channel_dense2",
        )(channel_update)
        return x + channel_update


class GELULayerNormMixerBlock(nn.Module):
    """HRM-style GELU Mixer with affine-optional post-LayerNorm.

    The two residual branches stay in the same order and normalization
    locations as the official SwiGLU/RMSNorm HRM Mixer.  Unlike
    :class:`GELUMixerBlock`, this HRM-style control deliberately has no
    additional ``tm_weights`` projection.
    """

    seq_len: int
    hidden_size: int
    token_hidden: int
    channel_hidden: int
    norm_eps: float = 1e-06
    layernorm_affine: bool = True
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if x.shape[-2:] != (self.seq_len, self.hidden_size):
            raise ValueError(
                f"Expected [..., {self.seq_len}, {self.hidden_size}], got {x.shape}."
            )
        token_input = jnp.swapaxes(x, -1, -2)
        token_update = nn.Dense(
            self.token_hidden,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="token_dense1",
        )(token_input)
        token_update = nn.gelu(token_update)
        token_update = nn.Dense(
            self.seq_len,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="token_dense2",
        )(token_update)
        token_output = nn.LayerNorm(
            epsilon=self.norm_eps,
            use_bias=self.layernorm_affine,
            use_scale=self.layernorm_affine,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            name="token_norm",
        )(token_input + token_update)
        x = jnp.swapaxes(token_output, -1, -2)
        channel_update = nn.Dense(
            self.channel_hidden,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="channel_dense1",
        )(x)
        channel_update = nn.gelu(channel_update)
        channel_update = nn.Dense(
            self.hidden_size,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            kernel_init=_comparison_init(),
            name="channel_dense2",
        )(channel_update)
        return nn.LayerNorm(
            epsilon=self.norm_eps,
            use_bias=self.layernorm_affine,
            use_scale=self.layernorm_affine,
            dtype=x.dtype,
            param_dtype=self.param_dtype,
            name="channel_norm",
        )(x + channel_update)


class RotaryEmbedding(nn.Module):
    """Stateless rotary embedding table used by the attention backbone."""

    dim: int
    max_position_embeddings: int
    base: float = 10000.0

    @nn.compact
    def __call__(self) -> CosSin:
        if self.dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {self.dim}.")
        indices = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / self.base ** (indices / self.dim)
        positions = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        frequencies = jnp.outer(positions, inv_freq)
        embeddings = jnp.concatenate([frequencies, frequencies], axis=-1)
        return (jnp.cos(embeddings), jnp.sin(embeddings))


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
                f"hidden_size must equal num_heads * head_dim; got {self.hidden_size}, {self.num_heads}, {self.head_dim}."
            )
        qkv = Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            batch_output_dims=(3,),
            param_dtype=self.param_dtype,
            name="qkv_proj",
        )(hidden_states)
        qkv = jnp.reshape(qkv, (*qkv.shape[:-1], self.num_heads, 3 * self.head_dim))
        query, key, value = jnp.split(qkv, 3, axis=-1)
        query = apply_rotary_pos_emb(query, cos_sin)
        key = apply_rotary_pos_emb(key, cos_sin)
        mask = None
        if self.is_causal:
            mask = nn.make_causal_mask(
                jnp.ones(hidden_states.shape[:-1], dtype=jnp.bool_)
            )
        output = nn.dot_product_attention(
            query, key, value, mask=mask, deterministic=True
        )
        output = jnp.reshape(output, (*output.shape[:-2], self.hidden_size))
        return Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            param_dtype=self.param_dtype,
            name="o_proj",
        )(output)


class SequenceBlock(nn.Module):
    """Official HRM post-norm attention or MLP-Mixer block."""

    seq_len: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    norm_eps: float = 1e-06
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array, cos_sin: Optional[CosSin] = None) -> Array:
        if x.shape[-2:] != (self.seq_len, self.hidden_size):
            raise ValueError(
                f"Expected [..., {self.seq_len}, {self.hidden_size}], got {x.shape}."
            )
        if self.is_mlp_mixer:
            token_input = jnp.swapaxes(x, -1, -2)
            token_update = SwiGLU(
                hidden_size=self.seq_len,
                intermediate_size=self.mlp_mixer_intermediate_size,
                param_dtype=self.param_dtype,
                name="mlp_t",
            )(token_input)
            x = jnp.swapaxes(
                rms_norm(token_input + token_update, self.norm_eps), -1, -2
            )
        else:
            if cos_sin is None:
                raise ValueError("cos_sin is required by the attention block.")
            attention_update = Attention(
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                num_heads=self.hidden_size // self.head_dim,
                is_causal=self.is_causal,
                param_dtype=self.param_dtype,
                name="attn",
            )(x, cos_sin)
            x = rms_norm(x + attention_update, self.norm_eps)
        if self.is_mlp_mixer:
            channel_update = SwiGLU(
                hidden_size=self.hidden_size,
                intermediate_size=self.mlp_mixer_intermediate_size,
                param_dtype=self.param_dtype,
                name="mlp",
            )(x)
        else:
            channel_update = AttentionFeedForward(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                param_dtype=self.param_dtype,
                name="mlp",
            )(x)
        return rms_norm(x + channel_update, self.norm_eps)


class BlockStack(nn.Module):
    """Stack of sequence blocks, independent of policy or task heads."""

    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    norm_eps: float = 1e-06
    rope_theta: float = 10000.0
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    block_type: Optional[str] = None
    token_hidden: int = 32
    channel_hidden: int = 32
    layernorm_affine: bool = True
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array) -> Array:
        resolved_block_type = _resolve_block_type(self.block_type, self.is_mlp_mixer)
        cos_sin = None
        if resolved_block_type == _ATTENTION_BLOCK_TYPE:
            cos_sin = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=self.seq_len,
                base=self.rope_theta,
                name="rotary_emb",
            )()
        output = hidden_states
        for layer_idx in range(self.num_layers):
            layer_name = f"layers_{layer_idx}"
            if resolved_block_type == "coghp_gelu":
                output = GELUMixerBlock(
                    seq_len=self.seq_len,
                    hidden_size=self.hidden_size,
                    token_hidden=self.token_hidden,
                    channel_hidden=self.channel_hidden,
                    is_causal=self.is_causal,
                    param_dtype=self.param_dtype,
                    name=layer_name,
                )(output)
            elif resolved_block_type == "gelu_layernorm":
                output = GELULayerNormMixerBlock(
                    seq_len=self.seq_len,
                    hidden_size=self.hidden_size,
                    token_hidden=self.token_hidden,
                    channel_hidden=self.channel_hidden,
                    norm_eps=self.norm_eps,
                    layernorm_affine=self.layernorm_affine,
                    param_dtype=self.param_dtype,
                    name=layer_name,
                )(output)
            else:
                output = SequenceBlock(
                    seq_len=self.seq_len,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    head_dim=self.head_dim,
                    is_causal=self.is_causal,
                    norm_eps=self.norm_eps,
                    is_mlp_mixer=resolved_block_type == "swiglu_rmsnorm",
                    mlp_mixer_intermediate_size=self.mlp_mixer_intermediate_size,
                    param_dtype=self.param_dtype,
                    name=layer_name,
                )(output, cos_sin=cos_sin)
        return output


class SequenceBackbone(nn.Module):
    """Shared-block action core with feed-forward, single-state, or HRM flow.

    ``L_level`` and ``H_level`` are always two independent instances of the
    same :class:`BlockStack` configuration.  Keeping those serialized names
    preserves existing HRM checkpoints; conceptually they are ``B_L`` and
    ``B_H`` for every ``core_type``.

    The single-state control starts from ``u_0 = 0``, injects the fixed input
    before every low update, and never injects it directly into a high update:
    ``u <- B_L(u + x)`` and ``u <- B_H(u)``.  Consequently H1L1 is exactly
    ``B_H(B_L(x))``, the same computation as ``matched_feedforward``.
    """

    seq_len: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    is_causal: bool
    h_cycles: int
    l_cycles: int
    core_type: str = "hrm"
    bptt: bool = True
    norm_eps: float = 1e-06
    rope_theta: float = 10000.0
    is_mlp_mixer: bool = False
    mlp_mixer_intermediate_size: int = 256
    block_type: Optional[str] = None
    token_hidden: int = 32
    channel_hidden: int = 32
    layernorm_affine: bool = True
    forward_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        if self.seq_len < 1 or self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("seq_len, hidden_size, and num_layers must be positive.")
        if self.h_cycles < 1 or self.l_cycles < 1:
            raise ValueError("h_cycles and l_cycles must both be positive.")
        if self.core_type not in BACKBONE_TYPES:
            raise ValueError(
                f"Unknown action core type {self.core_type!r}; expected one of {sorted(BACKBONE_TYPES)}."
            )
        resolved_block_type = _resolve_block_type(self.block_type, self.is_mlp_mixer)
        if self.core_type != "hrm" and resolved_block_type == _ATTENTION_BLOCK_TYPE:
            raise ValueError(
                f"matched_feedforward and single_state_iterative support only Mixer blocks {sorted(MIXER_BLOCK_TYPES)}."
            )
        if resolved_block_type == _ATTENTION_BLOCK_TYPE:
            if self.head_dim < 1 or self.hidden_size % self.head_dim != 0:
                raise ValueError(
                    "Attention HRM requires head_dim to divide hidden_size."
                )
        elif resolved_block_type == "swiglu_rmsnorm":
            if self.mlp_mixer_intermediate_size < 1:
                raise ValueError(
                    "mlp_mixer_intermediate_size must be positive for swiglu_rmsnorm."
                )
        elif self.token_hidden < 1 or self.channel_hidden < 1:
            raise ValueError(
                f"token_hidden and channel_hidden must be positive for {resolved_block_type}."
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
            block_type=self.block_type,
            token_hidden=self.token_hidden,
            channel_hidden=self.channel_hidden,
            layernorm_affine=self.layernorm_affine,
            param_dtype=self.param_dtype,
        )
        self.H_level = BlockStack(**level_kwargs, name="H_level")
        self.L_level = BlockStack(**level_kwargs, name="L_level")
        if self.core_type == "hrm":
            self.zH_init = self.variable(
                "constants",
                "zH_init",
                lambda: approximate_truncated_normal(1.0)(
                    self.make_rng("params"), (self.hidden_size,), self.forward_dtype
                ),
            )
            self.zL_init = self.variable(
                "constants",
                "zL_init",
                lambda: approximate_truncated_normal(1.0)(
                    self.make_rng("params"), (self.hidden_size,), self.forward_dtype
                ),
            )

    def initial_carry(self, batch_shape: Tuple[int, ...]) -> dict[str, Array]:
        """Create only the transient state required by the selected core."""
        shape = (*batch_shape, self.seq_len, self.hidden_size)
        if self.core_type == "matched_feedforward":
            return {}
        if self.core_type == "single_state_iterative":
            return {"u": jnp.zeros(shape, dtype=self.forward_dtype)}
        return {
            "z_H": jnp.broadcast_to(self.zH_init.value, shape),
            "z_L": jnp.broadcast_to(self.zL_init.value, shape),
        }

    def __call__(
        self, carry: Optional[Carry], inputs: Array
    ) -> tuple[dict[str, Array], Array]:
        x = inputs.astype(self.forward_dtype)
        if x.shape[-2:] != (self.seq_len, self.hidden_size):
            raise ValueError(
                f"Expected [..., {self.seq_len}, {self.hidden_size}], got {x.shape}."
            )
        if carry is None:
            carry = self.initial_carry(x.shape[:-2])
        if self.core_type == "matched_feedforward":
            low_output = self.L_level(x)
            output = self.H_level(low_output)
            return ({}, output)
        if self.core_type == "single_state_iterative":
            if "u" not in carry:
                raise ValueError(
                    "single_state_iterative carry must contain only state u."
                )
            u = carry["u"].astype(self.forward_dtype)
            if u.shape != x.shape:
                raise ValueError(
                    f"Single-state carry and input shapes must match; got x={x.shape}, u={u.shape}."
                )
            for iteration in range(self.h_cycles * self.l_cycles - 1):
                u = self.L_level(u + x)
                if (iteration + 1) % self.l_cycles == 0:
                    u = self.H_level(u)
            if not self.bptt:
                u = jax.lax.stop_gradient(u)
            u = self.L_level(u + x)
            u = self.H_level(u)
            return ({"u": jax.lax.stop_gradient(u)}, u)
        z_H = carry["z_H"].astype(self.forward_dtype)
        z_L = carry["z_L"].astype(self.forward_dtype)
        expected_shape = x.shape
        if z_H.shape != expected_shape or z_L.shape != expected_shape:
            raise ValueError(
                f"Carry and input shapes must match; got x={x.shape}, z_H={z_H.shape}, z_L={z_L.shape}."
            )
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
            "z_H": jax.lax.stop_gradient(z_H),
            "z_L": jax.lax.stop_gradient(z_L),
        }
        return (next_carry, z_H)


class ComparisonMixer(nn.Module):
    num_tokens: int
    embed_dim: int
    hidden_dim_tokens: int
    hidden_dim_channels: int
    init_scale: float = 0.01
    decay_alpha: float = 0.9
    causal: bool = True

    def setup(self):
        self.token_dense1 = nn.Dense(
            self.hidden_dim_tokens, kernel_init=_comparison_init()
        )
        self.token_dense2 = nn.Dense(self.num_tokens, kernel_init=_comparison_init())
        self.channel_dense1 = nn.Dense(
            self.hidden_dim_channels, kernel_init=_comparison_init()
        )
        self.channel_dense2 = nn.Dense(self.embed_dim, kernel_init=_comparison_init())
        self.tm_weights = self.param(
            "tm_weights",
            nn.initializers.normal(stddev=0.02),
            (self.num_tokens, self.num_tokens),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.transpose(x, (0, 2, 1))
        y = self.token_dense1(y)
        y = nn.gelu(y)
        y = self.token_dense2(y)
        y = jnp.transpose(y, (0, 2, 1))
        tm_weights = jnp.tril(self.tm_weights) if self.causal else self.tm_weights
        y = jnp.einsum("btd,ts->bsd", y, tm_weights)
        x = x + y
        z = self.channel_dense1(x)
        z = nn.gelu(z)
        z = self.channel_dense2(z)
        output = x + z
        return output


class SwiGLUBackbone(nn.Module):
    """A continuous-input backbone with residual SwiGLU and pre-RMSNorm.

    Numerical components stay private to this backbone layer; actors and values
    supply the input/output meaning. This is independent of CoGHP's GELU MLP.
    """

    hidden_dims: Sequence[int]
    out_dim: int
    expansion: int = 2
    norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        if not self.hidden_dims or self.out_dim < 1 or self.expansion < 1:
            raise ValueError(
                "Positive backbone widths, output size and expansion required."
            )
        for index, width in enumerate(self.hidden_dims):
            if width < 1:
                raise ValueError("Backbone widths must be positive.")
            if index == 0 or x.shape[-1] != width:
                x = nn.Dense(width, name=f"input_{index}")(x)
            update = SwiGLU(width, width * self.expansion, name=f"block_{index}")(
                rms_norm(x, self.norm_eps)
            )
            x = x + update
        return nn.Dense(self.out_dim, name="output")(rms_norm(x, self.norm_eps))
