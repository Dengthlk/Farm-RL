from __future__ import annotations

import math
from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from kaggriculture.actions import N_QUANTITIES, MarketKind, UnitAction
from kaggriculture.constants import BOARD_SIZE, QUANTITY_BINS


class ActorOutput(NamedTuple):
    unit_logits: Tensor
    market_kind_logits: Tensor
    market_quantity_context: Tensor


class Linear(nn.Linear):
    """Keep the bias epilogue identical across rollout and update batch sizes.

    A fused BF16 addmm can round the accumulator before or after adding bias,
    depending on the selected GEMM. Make that boundary explicit while retaining
    tensor-core matrix multiplication and FP32 optimizer master parameters.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        output = F.linear(inputs, self.weight, None)
        return output if self.bias is None else output + self.bias.to(output.dtype)


class RMSNorm(nn.RMSNorm):
    """Compute-dtype normalization with fixed CUDA forward reduction arithmetic.

    `scale`/`shift` of shape `(*inputs.shape[:-2], width)` apply the conditioning
    `(1 + scale) * normalized + shift`, broadcast over the token dimension. On
    CUDA the epilogue is fused into the normalization kernel.
    """

    def forward(
        self, inputs: Tensor, scale: Tensor | None = None, shift: Tensor | None = None
    ) -> Tensor:
        if inputs.is_cuda:
            from kaggriculture.triton_norm import rms_norm

            eps = self.eps if self.eps is not None else torch.finfo(torch.float32).eps
            return rms_norm(inputs, self.weight, eps, scale, shift)
        with torch.autocast(inputs.device.type, enabled=False):
            normalized = F.rms_norm(inputs, self.normalized_shape, self.weight, self.eps)
        if scale is None or shift is None:
            return normalized
        return normalized * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


class ReluSquared(nn.Module):
    """Parameter-free ReLU-squared activation used throughout the network.

    Written as a self-multiply rather than `.square()` because `aten::pow` sits on
    autocast's fp32 cast list while `aten::mul` does not. Under the production
    bf16 autocast `.square()` therefore upcast its input, ran in fp32, and returned
    fp32 -- on the FFN's 4x-wide activation, the widest tensor in the model. The
    precision bought nothing: every consumer is a `Linear` or `Conv2d`, which
    autocast casts straight back down, so both spellings round exactly once. The
    self-multiply keeps the activation in the compute dtype and halves its traffic.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        activated = F.relu(inputs)
        return activated * activated


class AxialRotaryEmbedding(nn.Module):
    """Real-valued, compile-friendly 2D RoPE for arbitrary board coordinates."""

    def __init__(
        self,
        head_dim: int,
        board_size: int = BOARD_SIZE,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("axial RoPE head width must be divisible by 4")
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
        )
        axis = torch.arange(board_size, dtype=torch.float32)
        axis_angles = torch.outer(axis, frequencies)
        axis_cos = axis_angles.cos().repeat_interleave(2, dim=-1)
        axis_sin = axis_angles.sin().repeat_interleave(2, dim=-1)
        y, x = torch.meshgrid(
            torch.arange(board_size),
            torch.arange(board_size),
            indexing="ij",
        )
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        self.board_size = board_size
        self.register_buffer(
            "cosine",
            torch.cat((axis_cos[flat_x], axis_cos[flat_y]), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "sine",
            torch.cat((axis_sin[flat_x], axis_sin[flat_y]), dim=-1),
            persistent=False,
        )

    @staticmethod
    def _rotate_pairs(inputs: Tensor) -> Tensor:
        pairs = inputs.unflatten(-1, (-1, 2))
        real, imaginary = pairs.unbind(dim=-1)
        return torch.stack((-imaginary, real), dim=-1).flatten(-2)

    def rotation(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Gather per-token cos/sin tables once for reuse by every layer.

        The gather output depends only on token positions, which are identical
        across all transformer layers, so hoisting it out of the attention
        modules removes the layer-count multiple of both the gather kernels and
        their saved activations.
        """
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("RoPE positions must have shape [batch, tokens, 2]")
        coordinates = positions.long().clamp(0, self.board_size - 1)
        indices = coordinates[..., 1] * self.board_size + coordinates[..., 0]
        batch, tokens = indices.shape
        cosine = self.cosine.index_select(0, indices.reshape(-1)).view(batch, 1, tokens, -1)
        sine = self.sine.index_select(0, indices.reshape(-1)).view_as(cosine)
        return cosine, sine

    @classmethod
    def apply_rotation(
        cls, query: Tensor, key: Tensor, rotation: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key shapes must match for self-attention RoPE")
        cosine, sine = rotation
        if cosine.shape[0] != query.shape[0] or cosine.shape[-2:] != query.shape[-2:]:
            raise ValueError("RoPE rotation does not match the query batch or token layout")
        cosine = cosine.to(dtype=query.dtype)
        sine = sine.to(dtype=query.dtype)
        return (
            query * cosine + cls._rotate_pairs(query) * sine,
            key * cosine + cls._rotate_pairs(key) * sine,
        )


def _sdpa_inputs(query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Use BF16 only inside CUDA SDPA so FP32 PPO heads remain replay-stable."""
    if query.device.type == "cuda":
        # CUDA Flash SDPA is generally unavailable for FP32. Keeping this cast
        # local avoids changing the CNN, residual stream, or policy head dtype.
        return query.bfloat16(), key.bfloat16(), value.bfloat16()
    return query, key, value


def initialize_policy_heads(
    unit_head: nn.Linear,
    market_kind: nn.Linear | None,
    market_quantity_context: nn.Linear,
    market_quantity_kind_gate: nn.Embedding,
    market_quantity_value: nn.Embedding,
    market_quantity_bias: nn.Parameter,
    *,
    action_interface: int = 1,
) -> None:
    """Initialize the shared policy heads every actor architecture uses."""
    for head in (unit_head, market_kind, market_quantity_context):
        if head is None:
            continue
        nn.init.normal_(head.weight, std=0.01)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
    nn.init.zeros_(market_quantity_kind_gate.weight)
    nn.init.normal_(market_quantity_value.weight, std=0.01)
    nn.init.zeros_(market_quantity_bias)

    # Legal masks already remove actions that cannot have an effect. Among
    # the remaining actions, favor completing an economic cycle over random
    # movement or destroying an investment. Every action remains learnable.
    with torch.no_grad():
        unit_bias = unit_head.bias
        unit_bias[UnitAction.PASS] = -1.25
        unit_bias[UnitAction.DROP] = 2.0
        for item, maximum, offset in (
            ("WHEAT", 16, 1.0),
            ("FERTILIZER", 8, 0.75),
            ("GOOSE", 4, 0.75),
            ("COW", 4, 0.75),
            ("SHEEP", 4, 0.75),
        ):
            for quantity in range(1, maximum + 1):
                unit_bias[UnitAction[f"PICKUP_{item}_{quantity}"]] = offset - math.log(quantity)
        unit_bias[UnitAction.PLACE_GOOSE : UnitAction.PLACE_SHEEP + 1] = 2.0
        unit_bias[UnitAction.PLANT_WHEAT : UnitAction.PLANT_MELON + 1] = 1.0
        unit_bias[UnitAction.WATER] = 2.0
        unit_bias[UnitAction.HARVEST] = 2.5
        unit_bias[UnitAction.FERTILIZE] = 1.0
        unit_bias[UnitAction.DIG] = -0.5
        unit_bias[UnitAction.BUILD_COOP : UnitAction.BUILD_PASTURE + 1] = -2.5
        unit_bias[UnitAction.FEED] = 2.0
        unit_bias[UnitAction.COLLECT_FERTILIZER] = 2.0
        unit_bias[UnitAction.CARE] = 1.0

        if action_interface == 3:
            # Row zero is a genuine no-order choice for every kind. Rows 1..100
            # are effective quantities and row 101 aliases the legal maximum.
            quantities = torch.arange(
                1,
                N_QUANTITIES + 1,
                device=market_quantity_bias.device,
                dtype=market_quantity_bias.dtype,
            )
            market_quantity_bias[:, 0] = 4.5
            market_quantity_bias[
                MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1, 1 : N_QUANTITIES + 1
            ].copy_(-2.0 * quantities.log())
            market_quantity_bias[
                MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_ANIMAL_SHEEP + 1,
                1 : N_QUANTITIES + 1,
            ].copy_(-2.5 * quantities.log())
            market_quantity_bias[
                MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1,
                1 : N_QUANTITIES + 1,
            ].copy_(0.5 * quantities.log())
            market_quantity_bias[MarketKind.BUY_LAND, 1] = -7.0
            return

        # Keep roughly 95% opening STOP probability and bias initial
        # exploration toward cheap hires/seeds and inventory liquidation.
        if market_kind is not None:
            kind_bias = market_kind.bias
            kind_bias[MarketKind.STOP] = 4.5
            kind_bias[MarketKind.HIRE] = 1.0
            kind_bias[MarketKind.BUY_LAND] = -7.0
            kind_bias[MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1] = -1.0
            kind_bias[MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_PRODUCT_FERTILIZER + 1] = -3.0
            kind_bias[MarketKind.BUY_ANIMAL_GOOSE : MarketKind.BUY_ANIMAL_SHEEP + 1] = -4.0
            kind_bias[MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1] = 4.0

        if action_interface == 4:
            # Four atoms (1, 2, 3, legal maximum), continuous weight, and
            # logistic location/scale.  A broad continuous component keeps
            # every legal integer reachable at initialization.
            market_quantity_bias[:, 4] = -0.5
            market_quantity_bias[:, 6] = -1.5
            market_quantity_bias[
                MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_ANIMAL_SHEEP + 1, 0
            ] = 1.5
            market_quantity_bias[MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1, 3] = 2.0
            return

        quantities = torch.as_tensor(
            QUANTITY_BINS,
            device=market_quantity_bias.device,
            dtype=market_quantity_bias.dtype,
        )
        log_quantity = quantities.log()
        market_quantity_bias[
            MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1, :N_QUANTITIES
        ].copy_(-2.0 * log_quantity)
        market_quantity_bias[
            MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_ANIMAL_SHEEP + 1, :N_QUANTITIES
        ].copy_(-2.5 * log_quantity)
        market_quantity_bias[
            MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1, :N_QUANTITIES
        ].copy_(0.5 * log_quantity)


def factored_quantity_logits(
    quantity_context: Tensor,
    market_kinds: Tensor,
    kind_gate: nn.Embedding,
    quantity_value: nn.Embedding,
    quantity_bias: Tensor,
    quantity_rank: int,
    quantity_mask: Tensor | None = None,
) -> Tensor:
    """Score the native sampler's small FP32 head, even under trunk autocast."""
    if quantity_context.shape[:-1] != market_kinds.shape:
        raise ValueError("quantity context and selected market kinds must align")
    if quantity_context.shape[-1] != quantity_rank:
        raise ValueError("quantity context has the wrong feature width")
    with torch.autocast(quantity_context.device.type, enabled=False):
        kinds = market_kinds.long()
        quantity_features = quantity_context.float() * (1.0 + kind_gate(kinds).float())
        # A matmul here may use TF32 under the trainer's global "high"
        # setting, even with autocast disabled. This small rank reduction is
        # fused by Inductor and follows the Rust sampler's FP32 accumulation.
        values = quantity_value.weight.float()
        scores = quantity_bias[kinds].float()
        for rank in range(quantity_rank):
            scores = scores + quantity_features[..., rank, None] * values[:, rank]
        if values.shape[0] == 7:
            if quantity_mask is None:
                raise ValueError("percentage quantity head requires a legality mask")
            return percentage_quantity_logits(scores, quantity_mask)
        if values.shape[0] == N_QUANTITIES:
            return scores
        if values.shape[0] != N_QUANTITIES + 1 or quantity_mask is None:
            raise ValueError("ALL quantity head requires a 100-bin legality mask")
        if quantity_mask.shape != (*market_kinds.shape, N_QUANTITIES):
            raise ValueError("quantity mask and selected market kinds must align")
        maximum = quantity_mask.long().sum(-1).sub(1).clamp_min(0)
        maximum_score = scores[..., :N_QUANTITIES].gather(-1, maximum[..., None])
        merged = torch.logaddexp(maximum_score, scores[..., N_QUANTITIES:])
        merged = torch.where(quantity_mask.any(-1, keepdim=True), merged, maximum_score)
        return scores[..., :N_QUANTITIES].scatter(-1, maximum[..., None], merged)


def percentage_quantity_logits(parameters: Tensor, mask: Tensor) -> Tensor:
    """Integer log masses of a logistic fraction plus 1/2/3/ALL atoms.

    For legal maximum m, amount q receives the logistic CDF mass between
    (q-1)/m and q/m. The component is conditioned to the interval [0, 1].
    Atoms that refer to the same executed amount are merged before sampling.
    The returned 100 logits can be masked and normalized by the usual exact
    categorical machinery in BC, PPO, and native inference.
    """
    if parameters.shape[-1] != 7 or mask.shape != (*parameters.shape[:-1], N_QUANTITIES):
        raise ValueError("percentage parameters and quantity mask must align")
    if parameters.dtype != torch.float32:
        parameters = parameters.float()
    maximum = mask.long().sum(-1).clamp_min(1)
    quantity = torch.arange(1, N_QUANTITIES + 1, device=parameters.device)
    scale = torch.nn.functional.softplus(parameters[..., 6:7]) + 0.02
    location = torch.sigmoid(parameters[..., 5:6])
    upper = (quantity / maximum[..., None] - location) / scale
    lower = ((quantity - 1) / maximum[..., None] - location) / scale
    delta = 1.0 / (maximum[..., None] * scale)
    # log(sigmoid(upper) - sigmoid(lower)), stable even in the tails.
    log_mass = (
        torch.nn.functional.logsigmoid(upper)
        + torch.nn.functional.logsigmoid(-lower)
        + torch.log(-torch.expm1(-delta))
    )
    high = (1.0 - location) / scale
    low = -location / scale
    log_total = (
        torch.nn.functional.logsigmoid(high)
        + torch.nn.functional.logsigmoid(-low)
        + torch.log(-torch.expm1(-1.0 / scale))
    )
    result = parameters[..., 4:5] + log_mass - log_total
    for atom, destination in enumerate((1, 2, 3, 0)):
        amount = maximum if atom == 3 else torch.full_like(maximum, destination)
        active = maximum >= destination if atom != 3 else mask.any(-1)
        index = (amount - 1).clamp(0, N_QUANTITIES - 1)[..., None]
        original = result.gather(-1, index)
        merged = torch.where(
            active[..., None],
            torch.logaddexp(original, parameters[..., atom : atom + 1]),
            original,
        )
        result = result.scatter(-1, index, merged)
    return result
