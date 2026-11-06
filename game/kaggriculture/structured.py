from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, NamedTuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from kaggriculture.constants import ANIMALS, BOARD_SIZE, CROPS, MAX_UNITS, PRODUCTS
from kaggriculture.model import AxialRotaryEmbedding, Linear, ReluSquared, RMSNorm, _sdpa_inputs
from kaggriculture.tokens import ANIMAL_PRIVATE_FIELDS, ANIMAL_TOKEN_FIELDS, CROP_PRIVATE_FIELDS, CROP_TOKEN_FIELDS, DEFAULT_OBSERVATION_SCHEMA_VERSION, N_TILE_CONTINUOUS, N_UNIT_CONTINUOUS, PRODUCT_TOKEN_FIELDS, QUADRANT_COUNT, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS, TILE_COUNT, TILE_KINDS, TILE_OCCUPANTS, UNIT_ROLES, UNIT_TILE_GATHERS, animal_token_fields, crop_token_fields, farm_token_fields, product_private_fields, product_token_fields, town_token_fields


@dataclass(frozen=True)
class StructuredConfig:
    """Target-architecture hyperparameters from docs/proposals/vit.md."""

    observation_schema_version: int = DEFAULT_OBSERVATION_SCHEMA_VERSION
    action_interface: int = 1
    model_dim: int = 128
    attention_heads: int = 4
    attention_kv_heads: int = 2
    ffn_multiplier: int = 2
    farm_blocks: int = 2
    opponent_latents: int = 8
    latents: int = 32
    core_layers: int = 8
    quantity_rank: int = 32
    # Empty means disabled. Research variants that want the canonical 8-layer
    # refresh schedule opt into ``global_refresh_context="all"`` explicitly;
    # an empty layer tuple must not silently enable extra blocks.
    global_refresh_layers: tuple[int, ...] = ()
    global_refresh_context: str = "none"
    input_reinject_layers: tuple[int, ...] = ()
    core_skip_source: int = 0
    core_skip_target: int = 0
    zero_init_branches: bool = False
    mudd_lite: bool = False
    fuse_market_decoder: bool = False
    fuse_unit_decoder: bool = False
    split_clock_token: bool = False
    global_modulation: bool = False
    fused_mlp: bool = False
    critic_core_layers: int = 0
    critic_latents: int = 0
    # Critic reads seed residual content from observations, not learned queries.
    critic_state_read: bool = False
    # Shared centralized decoder/readout, with one baseline per acting entity.
    per_entity_critic: bool = False
    # The actor's view of the opponent's board. Both farms run the shared
    # `farm_local` blocks and the opponent half reaches the rest of the actor
    # only as `opponent_summary`, so it is 30% of the actor's forward bytes at
    # production width -- half of `farm_local`, half of the tile embedding, and
    # the summary block. `scripts/ablate_opponent_farm.py` permutes that half
    # across a batch of real states and measures what the policy does about it:
    # mean unit KL 1.3e-5 and 0.04% of greedy actions changed at iteration 390,
    # against 3.80 and 59.9% for the same permutation of the actor's own farm.
    # The insensitivity is present at initialization and grows with training, so
    # it is a property of the architecture rather than a stage of learning.
    #
    # Disabling it is a learning change, not a speedup, and is off by default
    # until an A/B says otherwise: an actor that cannot see the opponent's board
    # cannot learn to react to it, and the ablation cannot distinguish "the
    # information is useless here" from "this path is too narrow to carry it".
    # The centralized critic is unaffected either way -- it reads both farms and
    # measurably uses them -- and the two trunks share no parameters.
    actor_opponent_farm: bool = True
    value_atoms: int = 255
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 3.0
    # Optional scalar ablation: half squared error without target clipping or
    # a readout softcap. The four value-support fields above are inert in it.
    scalar_value: bool = False

    def __post_init__(self) -> None:
        if self.action_interface not in (1, 2, 4):
            raise ValueError("action_interface must be 1, 2, or 4")
        if self.observation_schema_version not in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
            raise ValueError(
                "stale structured observation schema; fresh encoding and training required"
            )
        refresh_layers = tuple(self.global_refresh_layers)
        refresh_context = self.global_refresh_context
        if not refresh_layers and refresh_context == "all":
            refresh_layers = tuple(range(3, self.core_layers, 3))
            if not refresh_layers:
                refresh_context = "none"
        object.__setattr__(self, "global_refresh_layers", refresh_layers)
        object.__setattr__(self, "global_refresh_context", refresh_context)
        object.__setattr__(self, "input_reinject_layers", tuple(self.input_reinject_layers))
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.attention_heads <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("attention_heads must evenly divide model_dim")
        if (self.model_dim // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by 4 for axial RoPE")
        if (
            self.attention_kv_heads <= 0
            or self.attention_kv_heads > self.attention_heads
            or self.attention_heads % self.attention_kv_heads
        ):
            raise ValueError("attention KV heads must positively divide query heads")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if self.fused_mlp and (self.model_dim % 128 or self.model_dim * self.ffn_multiplier % 256):
            raise ValueError(
                "fused MLP requires model width divisible by 128 and hidden width by 256"
            )
        if self.farm_blocks <= 0:
            raise ValueError("farm_blocks must be positive")
        if self.opponent_latents <= 0:
            raise ValueError("opponent_latents must be positive")
        if self.latents <= 0:
            raise ValueError("latents must be positive")
        if self.core_layers <= 0:
            raise ValueError("core_layers must be positive")
        if self.quantity_rank <= 0:
            raise ValueError("quantity_rank must be positive")
        for name, layers in (
            ("global_refresh_layers", self.global_refresh_layers),
            ("input_reinject_layers", self.input_reinject_layers),
        ):
            if tuple(sorted(set(layers))) != layers:
                raise ValueError(f"{name} must be sorted and unique")
            if any(layer < 1 or layer > self.core_layers for layer in layers):
                raise ValueError(f"{name} must name layers in 1..core_layers")
        if self.global_refresh_context not in {"none", "economy", "all"}:
            raise ValueError("global_refresh_context must be none, economy, or all")
        if bool(self.global_refresh_layers) != (self.global_refresh_context != "none"):
            raise ValueError("global refresh layers and context must be enabled together")
        skip_enabled = bool(self.core_skip_source or self.core_skip_target)
        if skip_enabled and not (
            1 <= self.core_skip_source < self.core_skip_target <= self.core_layers
        ):
            raise ValueError("core skip must name an ordered source and target layer")
        if self.mudd_lite and self.core_layers < 6:
            raise ValueError("MUDD-lite needs at least six core layers")
        for name, value in (
            ("critic_core_layers", self.critic_core_layers),
            ("critic_latents", self.critic_latents),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.value_atoms < 2:
            raise ValueError("value_atoms must be at least 2")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max):
            raise ValueError("value support bounds must be finite")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not math.isfinite(self.value_sigma_ratio) or self.value_sigma_ratio <= 0:
            raise ValueError("value_sigma_ratio must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredInputs(NamedTuple):
    """Batched token tensors for one player viewpoint (own farm first)."""

    tile_categorical: Tensor  # [B, 2 * TILE_COUNT, 6] int64
    tile_continuous: Tensor  # [B, 2 * TILE_COUNT, N_TILE_CONTINUOUS]
    unit_categorical: Tensor  # [B, MAX_UNITS, 4] int64
    unit_continuous: Tensor  # [B, MAX_UNITS, N_UNIT_CONTINUOUS]
    unit_active: Tensor  # [B, MAX_UNITS] bool
    unit_tile_gather: Tensor  # [B, MAX_UNITS, 5] int64 into own-farm tiles
    unit_tile_gather_valid: Tensor  # [B, MAX_UNITS, 5] bool
    products: Tensor  # [B, len(PRODUCTS), product feature width]; models read their schema's prefix
    animals: Tensor  # [B, len(ANIMALS), animal feature width]
    crops: Tensor  # [B, len(CROPS), crop feature width]
    farms: Tensor  # [B, 2, len(FARM_TOKEN_FIELDS)]; models read their schema's prefix
    town: Tensor  # [B, len(TOWN_TOKEN_FIELDS)]


class CriticExtras(NamedTuple):
    """Opponent-side critic tensors stacked alongside StructuredInputs."""

    products: Tensor  # [B, len(PRODUCTS), len(PRODUCT_PRIVATE_FIELDS)]
    animals: Tensor  # [B, len(ANIMALS), len(ANIMAL_PRIVATE_FIELDS)]
    crops: Tensor  # [B, len(CROPS), len(CROP_PRIVATE_FIELDS)]
    unit_categorical: Tensor  # [B, MAX_UNITS, N_UNIT_CATEGORICAL] int64
    unit_continuous: Tensor  # [B, MAX_UNITS, N_UNIT_CONTINUOUS]
    unit_active: Tensor  # [B, MAX_UNITS] bool


def stack_structured(
    rows: list,
    device: torch.device | None = None,
) -> tuple[StructuredInputs, CriticExtras | None]:
    """Batch ``StructuredObservation`` bundles into model-ready tensors.

    Integer index arrays upcast to int64 for embedding lookups; continuous
    features keep their staged dtype (the embedders cast per forward). Critic
    extras are returned only when every row carries them.
    """
    if not rows:
        raise ValueError("cannot stack an empty batch")

    def stacked(field: str, dtype: torch.dtype | None = None) -> Tensor:
        tensor = torch.from_numpy(np.stack([getattr(row, field) for row in rows]))
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    inputs = StructuredInputs(
        tile_categorical=stacked("tile_categorical", torch.int64),
        tile_continuous=stacked("tile_continuous"),
        unit_categorical=stacked("unit_categorical", torch.int64),
        unit_continuous=stacked("unit_continuous"),
        unit_active=stacked("unit_active"),
        unit_tile_gather=stacked("unit_tile_gather", torch.int64),
        unit_tile_gather_valid=stacked("unit_tile_gather_valid"),
        products=stacked("products"),
        animals=stacked("animals"),
        crops=stacked("crops"),
        farms=stacked("farms"),
        town=stacked("town"),
    )
    with_extras = sum(row.critic_products is not None for row in rows)
    if with_extras == 0:
        return inputs, None
    if with_extras != len(rows):
        raise ValueError("critic extras must be present on every row or none")
    extras = CriticExtras(
        products=stacked("critic_products"),
        animals=stacked("critic_animals"),
        crops=stacked("critic_crops"),
        unit_categorical=stacked("opponent_unit_categorical", torch.int64),
        unit_continuous=stacked("opponent_unit_continuous"),
        unit_active=stacked("opponent_unit_active"),
    )
    return inputs, extras


class GatedResidual(nn.Module):
    """Near-identity residual: branch output scaled by a learned channel gate."""

    def __init__(self, width: int, initial: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.full((width,), initial))

    def forward(self, residual: Tensor, branch: Tensor) -> Tensor:
        compute_dtype = branch.dtype
        return residual.to(compute_dtype) + self.gate.to(compute_dtype) * branch


_FUSED_ATTENTION_HEAD_MULTIPLE = 8


SMALL_QUERY_INITIAL_SCALE = 0.02


def _pad_head_width(tensor: Tensor, width: int) -> Tensor:
    extra = width - tensor.shape[-1]
    return tensor if extra == 0 else nn.functional.pad(tensor, (0, extra))


def _masked_flex_attention(
    query: Tensor, key: Tensor, value: Tensor, key_valid: Tensor, scale: float
) -> Tensor:
    # Flex maps batch to CUDA grid Y, whose limit is 65535. Flattened per-unit
    # decoders exceed it at production minibatch sizes. Independent batch tiles
    # preserve the same fused operator and gradients without copying K/V.
    if query.shape[0] > 65535:
        outputs = []
        for start in range(0, query.shape[0], 32768):
            stop = start + 32768
            outputs.append(
                _masked_flex_attention(
                    query[start:stop],
                    key if key.shape[0] == 1 else key[start:stop],
                    value if value.shape[0] == 1 else value[start:stop],
                    key_valid if key_valid.shape[0] == 1 else key_valid[start:stop],
                    scale,
                )
            )
        return torch.cat(outputs, dim=0)

    def score_mod(score, b, h, q, k):
        mask_batch = 0 if key_valid.shape[0] == 1 else b
        return torch.where(key_valid[mask_batch, k], score, -float("inf"))

    return flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        scale=scale,
        # Default 128-row backward tiles overpad short folded entity queries.
        kernel_options={
            "BACKEND": "TRITON",
            **(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 64,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 64,
                }
                if query.shape[-1] <= 32 and query.shape[-2] <= 64 and key.shape[-2] <= 256
                else {}
            ),
        },
    )


def _fused_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    *,
    enable_gqa: bool,
    scale: float,
    backend: SDPBackend | None = None,
) -> Tensor:
    """One fused attention call, with the head width padded into kernel support.

    `scale` is stated rather than defaulted: SDPA derives its default from the
    padded width, which is not the width this attention is defined over.
    An explicit backend changes only the kernel; CUDA GQA still folds query
    groups without materializing repeated K/V heads.
    """
    head_dim = query.shape[-1]
    remainder = head_dim % _FUSED_ATTENTION_HEAD_MULTIPLE if query.is_cuda else 0
    if remainder:
        width = head_dim + _FUSED_ATTENTION_HEAD_MULTIPLE - remainder
        query, key, value = (_pad_head_width(t, width) for t in (query, key, value))
    fold_gqa = query.is_cuda and enable_gqa
    if fold_gqa:
        # Fold each query-head group into the sequence. Besides supporting equal-
        # head kernels, this avoids native GQA's expanded dK/dV intermediates.
        batch, heads, query_tokens, width = query.shape
        kv_heads = key.shape[1]
        grouped_tokens = (heads // kv_heads) * query_tokens
        query = query.reshape(batch, kv_heads, grouped_tokens, width)
        # Key-only masks already broadcast over the folded query sequence.
        # Query/head-specific masks must follow the original score layout.
        if (
            attention_mask is not None
            and attention_mask.ndim >= 2
            and (
                attention_mask.shape[-2] != 1
                or (attention_mask.ndim >= 3 and attention_mask.shape[-3] != 1)
            )
        ):
            attention_mask = attention_mask.expand(
                batch, heads, query_tokens, key.shape[-2]
            ).reshape(batch, kv_heads, grouped_tokens, key.shape[-2])
        enable_gqa = False
    if (
        backend is None
        and fold_gqa
        and query.dtype in (torch.float16, torch.bfloat16)
        and torch.is_grad_enabled()
        and attention_mask is not None
        and attention_mask.dtype == torch.bool
        and attention_mask.ndim == 4
        and attention_mask.shape[1:3] == (1, 1)
    ):
        # Fuse validity into dense scores; constructing sparse block metadata
        # does not pay for these short sequences. The decoding heuristic is
        # slower here. No-grad rollout retains SDPA's supported vmap batching.
        attended = _masked_flex_attention(query, key, value, attention_mask[:, 0, 0, :], scale)
    else:
        # Prefer Flash for dense attention and cuDNN for arbitrary masks. Keep
        # efficient CUDA for unsupported configurations, never a math fallback.
        # Explicit benchmark backends remain exclusive and fail loudly.
        backends: SDPBackend | list[SDPBackend]
        if backend is not None:
            backends = backend
        elif query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            preferred = (
                SDPBackend.FLASH_ATTENTION if attention_mask is None else SDPBackend.CUDNN_ATTENTION
            )
            backends = [preferred, SDPBackend.EFFICIENT_ATTENTION]
        else:
            backends = SDPBackend.EFFICIENT_ATTENTION if query.is_cuda else SDPBackend.MATH
        with sdpa_kernel(backends, set_priority=True):
            attended = nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=scale,
                enable_gqa=enable_gqa,
            )
    if fold_gqa:
        attended = attended.reshape(batch, heads, query_tokens, attended.shape[-1])
    return attended[..., :head_dim] if remainder else attended


class Attention(nn.Module):
    """QK-normalized attention over an explicit context (self or cross)."""

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.attention_kv_heads
        self.head_dim = config.model_dim // self.heads
        self.query = Linear(config.model_dim, config.model_dim, bias=False)
        self.key_value = Linear(config.model_dim, 2 * self.kv_heads * self.head_dim, bias=False)
        self.query_norm = RMSNorm(self.head_dim)
        self.key_norm = RMSNorm(self.head_dim)
        self.output = Linear(config.model_dim, config.model_dim, bias=False)
        if config.zero_init_branches:
            nn.init.zeros_(self.output.weight)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        *,
        query_rotation: tuple[Tensor, Tensor] | None = None,
        key_rotation: tuple[Tensor, Tensor] | None = None,
        context_valid: Tensor | None = None,
    ) -> Tensor:
        batch, query_tokens, width = queries.shape
        key_tokens = context.shape[1]
        query = (
            self.query(queries).view(batch, query_tokens, self.heads, self.head_dim).transpose(1, 2)
        )
        key, value = (
            self.key_value(context)
            .view(batch, key_tokens, 2, self.kv_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        query = self.query_norm(query)
        key = self.key_norm(key)
        if query_rotation is not None:
            cosine, sine = query_rotation
            query = query * cosine.to(query.dtype) + AxialRotaryEmbedding._rotate_pairs(
                query
            ) * sine.to(query.dtype)
        if key_rotation is not None:
            cosine, sine = key_rotation
            key = key * cosine.to(key.dtype) + AxialRotaryEmbedding._rotate_pairs(key) * sine.to(
                key.dtype
            )
        attention_mask = None
        if context_valid is not None:
            attention_mask = context_valid.view(batch, 1, 1, key_tokens)
        query, key, value = _sdpa_inputs(query, key, value)
        scale = self.head_dim**-0.5
        attended = _fused_attention(
            query,
            key,
            value,
            attention_mask,
            enable_gqa=self.heads != self.kv_heads,
            scale=scale,
        )
        attended = attended.to(dtype=queries.dtype)
        return self.output(attended.transpose(1, 2).reshape(batch, query_tokens, width))


class FeedForward(nn.Module):
    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        hidden = config.model_dim * config.ffn_multiplier
        self.input = Linear(config.model_dim, hidden)
        self.activation = ReluSquared()
        self.output = Linear(hidden, config.model_dim)
        if config.zero_init_branches:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.activation(self.input(inputs)))


class Block(nn.Module):
    """Pre-norm block; state reads initialize residuals from normalized attention."""

    def __init__(
        self,
        config: StructuredConfig,
        *,
        residual_initial: float = 0.1,
        conditioned: bool = False,
        state_read: bool = False,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.model_dim)
        # A state read is the input projection, not a zero-initialized residual branch.
        attention_config = replace(config, zero_init_branches=False) if state_read else config
        self.attention = Attention(attention_config)
        self.attention_gate = (
            None if state_read else GatedResidual(config.model_dim, residual_initial)
        )
        self.read_norm = RMSNorm(config.model_dim, elementwise_affine=False) if state_read else None
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim, residual_initial)
        self.modulation = Linear(config.model_dim, 4 * config.model_dim) if conditioned else None
        if self.modulation is not None:
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        queries: Tensor,
        context: Tensor | None = None,
        *,
        context_norm: nn.Module | None = None,
        query_rotation: tuple[Tensor, Tensor] | None = None,
        key_rotation: tuple[Tensor, Tensor] | None = None,
        context_valid: Tensor | None = None,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        if self.read_norm is not None and context is None:
            raise ValueError("a state read requires observation context")
        attention_input = self.attention_norm(queries)
        ffn_scale = ffn_shift = None
        if self.modulation is not None:
            if conditioning is None:
                raise ValueError("conditioned block requires one vector per batch row")
            attention_scale, attention_shift, ffn_scale, ffn_shift = (
                self.modulation(conditioning).to(attention_input.dtype).chunk(4, dim=-1)
            )
            attention_input = attention_input * (
                1 + attention_scale.unsqueeze(1)
            ) + attention_shift.unsqueeze(1)
        elif conditioning is not None:
            raise ValueError("unconditioned block does not accept conditioning")
        if context is None:
            keys = attention_input
        else:
            keys = context_norm(context) if context_norm is not None else context
        hidden = self.attention(
            attention_input,
            keys,
            query_rotation=query_rotation,
            key_rotation=key_rotation,
            context_valid=context_valid,
        )
        if self.read_norm is not None:
            hidden = self.read_norm(hidden)
        else:
            assert self.attention_gate is not None
            hidden = self.attention_gate(queries, hidden)
        ffn_input = self.ffn_norm(hidden)
        if ffn_scale is not None and ffn_shift is not None:
            ffn_input = ffn_input * (1 + ffn_scale.unsqueeze(1)) + ffn_shift.unsqueeze(1)
        return self.ffn_gate(hidden, self.ffn(ffn_input))


class _TinyVocabularyEmbedding(torch.autograd.Function):
    """Sum of embedding lookups whose tables have only a handful of rows.

    The forward is the ordinary gather. The backward is one dense GEMM of the
    transposed one-hot indices against the gradient: an embedding table this
    small turns the stock scatter-add backward into millions of atomic adds
    contending for a few hundred addresses, which at 1.6M tile tokens per
    minibatch cost more than the whole farm attention backward.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(indices: Tensor, *weights: Tensor) -> Tensor:
        embedded = weights[0][indices[..., 0]]
        for column, weight in enumerate(weights[1:], start=1):
            embedded = embedded + weight[indices[..., column]]
        return embedded

    @staticmethod
    def setup_context(ctx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        indices, *weights = inputs
        ctx.save_for_backward(indices)
        ctx.rows = tuple(weight.shape[0] for weight in weights)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor | None, ...]:
        (indices,) = ctx.saved_tensors
        columns = len(ctx.rows)
        flat_gradient = gradient.reshape(-1, gradient.shape[-1])
        # One combined one-hot over the concatenated vocabularies: each token
        # row carries one 1 per table, so a single GEMM yields every table's
        # gradient stacked along the rows.
        offsets = torch.tensor(
            tuple(sum(ctx.rows[:column]) for column in range(columns)),
            device=indices.device,
            dtype=indices.dtype,
        )
        one_hot = nn.functional.one_hot(
            (indices.reshape(-1, columns) + offsets).reshape(-1), sum(ctx.rows)
        )
        one_hot = one_hot.view(-1, columns, sum(ctx.rows)).sum(dim=1).to(gradient.dtype)
        stacked = one_hot.t() @ flat_gradient
        return None, *stacked.split(ctx.rows, dim=0)


class TileEmbedder(nn.Module):
    """Categorical embeddings plus a bounded-continuous MLP per tile token.

    Tile tokens are laid out own farm first, ``y * BOARD_SIZE + x`` within a farm,
    so the farm, row, column, and quadrant columns are functions of the token slot
    alone (`tokens.TILE_SLOT_CATEGORICAL`). Their embeddings are therefore one
    ``[tokens, width]`` table added to every row instead of four per-token
    lookups, and their backward is a batch reduction rather than a scatter.
    """

    def __init__(self, config: StructuredConfig) -> None:
        super().__init__()
        width = config.model_dim
        self.kind = nn.Embedding(len(TILE_KINDS), width)
        self.occupant = nn.Embedding(len(TILE_OCCUPANTS), width)
        self.farm = nn.Embedding(2, width)
        self.row = nn.Embedding(BOARD_SIZE, width)
        self.column = nn.Embedding(BOARD_SIZE, width)
        self.quadrant = nn.Embedding(QUADRANT_COUNT, width)
        self.continuous = nn.Sequential(
            Linear(N_TILE_CONTINUOUS, width),
            ReluSquared(),
            Linear(width, width),
        )

    def slot_embedding(self, tokens: int, device: torch.device) -> Tensor:
        """Farm/row/column/quadrant embeddings for the leading ``tokens`` slots."""
        token = torch.arange(tokens, device=device)
        farm, position = token // TILE_COUNT, token % TILE_COUNT
        row, column = position // BOARD_SIZE, position % BOARD_SIZE
        half = BOARD_SIZE // 2
        quadrant = (row >= half).long() * 2 + (column >= half).long()
        return (
            self.farm.weight[farm]
            + self.row.weight[row]
            + self.column.weight[column]
            + self.quadrant.weight[quadrant]
        )

    def forward(self, categorical: Tensor, continuous: Tensor) -> Tensor:
        vocabulary = categorical[..., :2]
        weights = (self.kind.weight, self.occupant.weight)
        # The custom backward is only for a table that receives a gradient. A
        # frozen copy run under grad mode takes the plain gather, which Dynamo
        # also traces where it cannot trace this vararg Function with no input
        # requiring grad.
        embedded = (
            _TinyVocabularyEmbedding.apply(vocabulary, *weights)
            if torch.is_grad_enabled() and any(weight.requires_grad for weight in weights)
            else _TinyVocabularyEmbedding.forward(vocabulary, *weights)
        )
        projected = self.continuous(continuous.to(self.continuous[0].weight.dtype))
        slots = self.slot_embedding(categorical.shape[-2], categorical.device)
        # Enter the compute dtype here: the residual stream is already in it after
        # the first gated residual, so this only moves one rounding earlier while
        # halving the traffic and saved activations of the first farm block.
        return (embedded + slots + projected).to(projected.dtype)


class UnitEmbedder(nn.Module):
    """Unit identity, execution slot, position, inventory, and local tiles."""

    def __init__(
        self,
        config: StructuredConfig,
        *,
        local_init: bool = True,
        local_context: bool = True,
    ) -> None:
        super().__init__()
        width = config.model_dim
        self.role = nn.Embedding(len(UNIT_ROLES), width)
        self.slot = nn.Embedding(MAX_UNITS, width)
        self.row = nn.Embedding(BOARD_SIZE, width)
        self.column = nn.Embedding(BOARD_SIZE, width)
        self.farm = nn.Embedding(2, width)
        self.continuous = nn.Sequential(
            Linear(N_UNIT_CONTINUOUS, width),
            ReluSquared(),
            Linear(width, width),
        )
        self.gather_relation = (
            nn.Embedding(len(UNIT_TILE_GATHERS), width) if local_init or local_context else None
        )
        self.gather_projection = (
            Linear(len(UNIT_TILE_GATHERS) * width, width, bias=False) if local_init else None
        )

    def local_tiles(self, farm_tiles: Tensor, gather: Tensor, gather_valid: Tensor) -> Tensor:
        """Gather each unit's HERE/NSEW encoded tiles: [B, U, 5, width]."""
        if self.gather_relation is None:
            raise ValueError("unit embedder was constructed without local context")
        batch, units, slots = gather.shape
        width = farm_tiles.shape[-1]
        flat = gather.reshape(batch, units * slots, 1).expand(-1, -1, width)
        local = farm_tiles.gather(1, flat).view(batch, units, slots, width)
        local = torch.where(gather_valid.unsqueeze(-1), local, 0.0)
        return local + self.gather_relation.weight

    def forward(
        self,
        categorical: Tensor,
        continuous: Tensor,
        active: Tensor,
        local_tiles: Tensor | None,
        *,
        opponent: bool,
    ) -> Tensor:
        embedded = (
            self.role(categorical[..., 0])
            + self.slot(categorical[..., 1])
            + self.row(categorical[..., 2])
            + self.column(categorical[..., 3])
            + self.farm.weight[int(opponent)]
            + self.continuous(continuous.to(self.continuous[0].weight.dtype))
        )
        if self.gather_projection is not None:
            if local_tiles is None:
                raise ValueError("local unit initialization requires gathered tiles")
            batch, units, slots, width = local_tiles.shape
            embedded = embedded + self.gather_projection(
                local_tiles.reshape(batch, units, slots * width)
            )
        return torch.where(active.unsqueeze(-1), embedded, 0.0)


def _schema_columns(tokens: Tensor, public: int, staged: int, private: int) -> Tensor:
    """A schema's public prefix of staged tokens, then its private prefix if any.

    ``staged`` is the newest schema's public width, where a critic's appended
    private columns begin. The result is contiguous, the layout an older
    schema's projection always read, so its kernels see the same operand.
    """
    if not private:
        return tokens[..., :public].contiguous()
    return torch.cat((tokens[..., :public], tokens[..., staged : staged + private]), dim=-1)


class EconomyEmbedder(nn.Module):
    """Product, crop, farm-summary, and town tokens with identity embeddings."""

    def __init__(self, config: StructuredConfig, *, private_columns: bool) -> None:
        super().__init__()
        width = config.model_dim
        schema = config.observation_schema_version
        # Staged product, animal and crop tokens carry the newest schema's
        # columns and, in a critic's input, the newest private columns after
        # them; this model reads the prefix of each its own schema defines.
        self.product_width = len(product_token_fields(schema))
        self.product_private_width = len(product_private_fields(schema)) if private_columns else 0
        product_width = self.product_width + self.product_private_width
        self.animal_width = len(animal_token_fields(schema))
        self.animal_private_width = len(ANIMAL_PRIVATE_FIELDS) if private_columns else 0
        animal_width = self.animal_width + self.animal_private_width
        self.crop_width = len(crop_token_fields(schema))
        self.crop_private_width = len(CROP_PRIVATE_FIELDS) if private_columns else 0
        crop_width = self.crop_width + self.crop_private_width
        self.private_columns = private_columns
        self.split_clock = config.split_clock_token
        self.product_identity = nn.Embedding(len(PRODUCTS), width)
        self.product_projection = Linear(product_width, width)
        self.animal_identity = nn.Embedding(len(ANIMALS), width)
        self.animal_projection = Linear(animal_width, width)
        self.crop_identity = nn.Embedding(len(CROPS), width)
        self.crop_projection = Linear(crop_width, width)
        self.farm_identity = nn.Embedding(2, width)
        # Staged farm and town tokens carry the newest schema's columns; this
        # model reads the prefixes its own schema defines, so v3 weights see v3
        # inputs.
        self.farm_width = len(farm_token_fields(config.observation_schema_version))
        self.farm_projection = Linear(self.farm_width, width)
        self.town_width = len(town_token_fields(config.observation_schema_version))
        if self.split_clock:
            self.clock_projection = Linear(6, width)
            self.town_projection = Linear(self.town_width - 6, width)
        else:
            self.clock_projection = None
            self.town_projection = Linear(self.town_width, width)

    def forward(
        self, products: Tensor, animals: Tensor, crops: Tensor, farms: Tensor, town: Tensor
    ) -> Tensor:
        dtype = self.product_projection.weight.dtype
        products = _schema_columns(
            products, self.product_width, len(PRODUCT_TOKEN_FIELDS), self.product_private_width
        )
        animals = _schema_columns(
            animals, self.animal_width, len(ANIMAL_TOKEN_FIELDS), self.animal_private_width
        )
        crops = _schema_columns(
            crops, self.crop_width, len(CROP_TOKEN_FIELDS), self.crop_private_width
        )
        tokens = [
            self.product_projection(products.to(dtype)) + self.product_identity.weight,
            self.animal_projection(animals.to(dtype)) + self.animal_identity.weight,
            self.crop_projection(crops.to(dtype)) + self.crop_identity.weight,
            self.farm_projection(farms[..., : self.farm_width].to(dtype))
            + self.farm_identity.weight,
        ]
        town = town[..., : self.town_width]
        if self.clock_projection is None:
            tokens.append(self.town_projection(town.to(dtype)).unsqueeze(1))
        else:
            tokens.extend(
                (
                    self.clock_projection(town[..., :6].to(dtype)).unsqueeze(1),
                    self.town_projection(town[..., 6:].to(dtype)).unsqueeze(1),
                )
            )
        return torch.cat(tokens, dim=1)

    def widened_state(self, source: EconomyEmbedder) -> dict[str, Tensor]:
        """``source``'s parameters laid out for this embedder's newer schema.

        Every schema appends its columns (the private ones after the public
        prefix), so each of ``source``'s input columns has a place here. Its
        weights move there and every column only this schema reads gets zero
        weight; the raw columns meet nothing before the projection, so the
        widened embedder computes ``source``'s embedding plus exact zeros on
        any staged tokens. GEMM kernels may still reassociate the older terms
        differently for a wider operand, so equality is exact in real
        arithmetic and to rounding in floating point.
        """
        if (self.private_columns, self.split_clock) != (source.private_columns, source.split_clock):
            raise ValueError("a schema upgrade cannot change the embedder's token layout")

        def placed(public: int, private: int, widened_public: int) -> list[int]:
            # The source's public prefix, then its private one after ours.
            return [*range(public), *range(widened_public, widened_public + private)]

        columns = {
            "product_projection": placed(
                source.product_width, source.product_private_width, self.product_width
            ),
            "animal_projection": placed(
                source.animal_width, source.animal_private_width, self.animal_width
            ),
            "crop_projection": placed(
                source.crop_width, source.crop_private_width, self.crop_width
            ),
            "farm_projection": range(source.farm_width),
            # With a split clock this projection reads the columns after the six
            # clock ones, which are a prefix there too.
            "town_projection": range(source.town_projection.in_features),
        }
        state = source.state_dict()
        for name, placement in columns.items():
            weight = state[f"{name}.weight"]
            widened = weight.new_zeros(getattr(self, name).weight.shape)
            widened[:, list(placement)] = weight
            state[f"{name}.weight"] = widened
        return state


def _token_mean(tokens: Tensor) -> Tensor:
    """Keep conditioning reduction order fixed across rollout/update batches.

    A batch-dependent parallel FP32 reduction can cross a BF16 midpoint before
    the modulation projection. The short token axis is accumulated in order;
    Inductor fuses the additions without allocating intermediate tensors.
    """
    total = tokens.select(-2, 0).float()
    for token in range(1, tokens.shape[-2]):
        total = total + tokens.select(-2, token).float()
    return (total / tokens.shape[-2]).to(tokens.dtype)


class StructuredDecisionBelief(NamedTuple):
    """Normalized entity-decoder outputs consumed by the policy heads."""

    unit_decisions: Tensor
    market_decisions: Tensor


class JepaBelief(NamedTuple):
    """Everything the shared `lejepa` backbone hands to its one world-model arm.

    The first two fields are the per-slot decision states; the last two are the
    encoded observation, which is what makes the objective a world model rather
    than a policy-readout regularizer. Under the shared backbone this is also the
    entire interface between the world model and its two consumers: the actor's
    heads and the critic's private tower both read these four tensors and nothing
    else, and both read them detached.
    """

    unit_decisions: Tensor  # (B, MAX_UNITS, D)
    market_decisions: Tensor  # (B, MAX_MARKET_ORDERS, D)
    economy: Tensor  # (B, economy tokens, D)
    tiles: Tensor  # (B, 2 * TILE_COUNT, D), own farm then opponent farm

    def detach(self) -> JepaBelief:
        """The same latents with the gradient cut, which is how consumers read them.

        The backbone is trained by `lejepa.jepa_horizon_loss` alone. Every path
        from a policy or value objective back into it passes through here, so the
        detach is one call in one place rather than a convention each consumer
        has to remember.
        """
        return JepaBelief(*(value.detach() for value in self))
