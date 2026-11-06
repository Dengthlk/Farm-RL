from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from kaggriculture.constants import CROPS, MAX_MARKET_ORDERS, MAX_UNITS, QUANTITY_BINS
from kaggriculture.orientation import Orientation, movement_permutation
from kaggriculture.resource_conditioning import RESOURCE_FEATURES, market_resource_features
from kaggriculture.tokens import encode_structured_observation
from kaggriculture.structured import stack_structured
from kaggriculture.lejepa_model import LejepaActor
from kaggriculture.actions import N_MARKET_KINDS, N_QUANTITIES, N_UNIT_ACTIONS, QUANTIFIED_MARKET_KINDS, MarketKind, MarketLedger, UnitAction, _apply_ledger_order, _ledger_kind_mask, _ledger_quantity_mask, apply_unit_shed_effect, apply_unit_tile_effect, compile_action, copy_tile_grid, unit_action_mask


@dataclass(frozen=True)
class PreparedQuantityHeads:
    """Immutable CPU quantity parameters for a frozen inference actor."""

    kind_gate: np.ndarray
    values: np.ndarray
    bias: np.ndarray
    resource_kind: np.ndarray | None = None
    resource_quantity: np.ndarray | None = None


def prepare_quantity_heads(
    actor: LejepaActor,
) -> PreparedQuantityHeads:
    """Materialize quantity parameters once for repeated frozen-policy actions."""

    def frozen(parameter: Tensor) -> np.ndarray:
        array = parameter.detach().float().cpu().numpy().copy()
        array.setflags(write=False)
        return array

    return PreparedQuantityHeads(
        kind_gate=frozen(actor.market_quantity_kind_gate.weight),
        values=frozen(actor.market_quantity_value.weight),
        bias=frozen(actor.market_quantity_bias),
        resource_kind=frozen(actor.market_resource_conditioner.kind.weight)
        if getattr(actor, "market_resource_conditioner", None) is not None
        else None,
        resource_quantity=frozen(actor.market_resource_conditioner.quantity.weight)
        if getattr(actor, "market_resource_conditioner", None) is not None
        else None,
    )


def percentage_quantity_logits_numpy(parameters: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Reference FP32 integer logits for the percentage quantity interface."""
    if parameters.shape[-1] != 7 or masks.shape != (*parameters.shape[:-1], N_QUANTITIES):
        raise ValueError("percentage parameters and quantity masks must align")
    maximum = np.maximum(masks.sum(axis=-1), 1).astype(np.int64)
    quantity = np.arange(1, N_QUANTITIES + 1, dtype=np.float32)
    scale = np.logaddexp(np.float32(0), parameters[..., 6:7]) + np.float32(0.02)
    location = np.float32(1) / (np.float32(1) + np.exp(-parameters[..., 5:6]))
    upper = (quantity / maximum[..., None] - location) / scale
    lower = ((quantity - 1) / maximum[..., None] - location) / scale
    delta = np.float32(1) / (maximum[..., None] * scale)

    def log_sigmoid(value: np.ndarray) -> np.ndarray:
        return -np.logaddexp(np.float32(0), -value)

    log_mass = log_sigmoid(upper) + log_sigmoid(-lower) + np.log(-np.expm1(-delta))
    high = (np.float32(1) - location) / scale
    low = -location / scale
    log_total = log_sigmoid(high) + log_sigmoid(-low) + np.log(-np.expm1(-np.float32(1) / scale))
    result = parameters[..., 4:5] + log_mass - log_total
    for atom, amount in enumerate((1, 2, 3, 0)):
        destinations = maximum if atom == 3 else np.full_like(maximum, amount)
        active = masks.any(-1) if atom == 3 else maximum >= amount
        indices = (destinations - 1).clip(0, N_QUANTITIES - 1)
        row = np.arange(indices.size)
        flat = result.reshape(-1, N_QUANTITIES)
        old = flat[row, indices.reshape(-1)]
        flat[row, indices.reshape(-1)] = np.where(
            active.reshape(-1),
            np.logaddexp(old, parameters[..., atom].reshape(-1)),
            old,
        )
    return result


def _sample_numpy_categorical(
    logits: np.ndarray,
    mask: np.ndarray,
    deterministic: bool,
    temperature: float,
    generator: np.random.Generator,
    *,
    draws: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} != {mask.shape}")
    if not mask.any(axis=-1).all():
        raise ValueError("every categorical decision needs at least one valid action")
    masked = np.where(mask, logits / max(temperature, 1e-4), -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    positive_mass = weights > 0
    if deterministic:
        total = weights.sum(axis=-1, keepdims=True, dtype=np.float64)
        actions = masked.argmax(axis=-1)
    else:
        cumulative = np.cumsum(weights, axis=-1, dtype=np.float64)
        total = cumulative[:, -1:]
        if draws is None:
            draws = generator.random((weights.shape[0], 1))
        elif draws.shape != (weights.shape[0], 1):
            raise ValueError(
                f"categorical draw shape mismatch: {draws.shape} != {(weights.shape[0], 1)}"
            )
        intervals = positive_mass & (draws * total < cumulative)
        last_positive = weights.shape[-1] - 1 - positive_mass[:, ::-1].argmax(axis=-1)
        actions = np.where(intervals.any(axis=-1), intervals.argmax(axis=-1), last_positive)
    log_total = np.log(total[:, 0])
    logprobs = shifted[np.arange(weights.shape[0]), actions] - log_total
    # Zero-mass entries contribute zero even when their masked logit is -inf.
    np.multiply(weights, shifted, out=weights, where=positive_mass)
    entropy = log_total - weights.sum(axis=-1, dtype=np.float64) / total[:, 0]
    return actions, logprobs.astype(np.float32), entropy.astype(np.float32)


@torch.inference_mode()
def choose_actions(actor, observations, quantity_heads):
    deterministic = True
    temperature = 1.0
    orientation = Orientation.IDENTITY
    generator = np.random.default_rng()
    encoded = [encode_structured_observation(obs, None) for obs in observations]
    inputs, _ = stack_structured(encoded, device=next(actor.parameters()).device)
    output = actor(inputs)
    unit_logits = output.unit_logits.float().cpu().numpy()
    market_kind_logits = output.market_kind_logits.float().cpu().numpy()
    market_quantity_context = output.market_quantity_context.float().cpu().numpy()
    if quantity_heads is None:
        quantity_heads = prepare_quantity_heads(actor)
    quantity_kind_gate = quantity_heads.kind_gate
    quantity_values = quantity_heads.values
    quantity_bias = quantity_heads.bias
    batch_size = len(observations)
    movement_map = movement_permutation(orientation)

    unit_actions = np.zeros((batch_size, MAX_UNITS), dtype=np.int64)
    unit_masks = np.zeros((batch_size, MAX_UNITS, N_UNIT_ACTIONS), dtype=np.bool_)
    real_unit_actions = np.zeros((batch_size, MAX_UNITS), dtype=np.int64)
    unit_active = np.stack([row.unit_active for row in encoded])
    remaining_seeds = [
        dict((observation.get("private") or {}).get("seeds") or {}) for observation in observations
    ]
    remaining_unit_sheds = [
        dict((observation.get("private") or {}).get("shed") or {}) for observation in observations
    ]
    unit_tiles = [
        copy_tile_grid(
            (observation.get("farms") or [])[int(observation.get("player", 0) or 0)].get("tiles")
            or []
        )
        for observation in observations
    ]
    unit_logprobs = np.zeros((batch_size, MAX_UNITS), dtype=np.float32)
    unit_entropies = np.zeros((batch_size, MAX_UNITS), dtype=np.float32)
    for unit_index in range(MAX_UNITS):
        for row, observation in enumerate(observations):
            if unit_active[row, unit_index]:
                unit_masks[row, unit_index] = unit_action_mask(
                    observation,
                    unit_index,
                    remaining_seeds[row],
                    remaining_unit_sheds[row],
                    unit_tiles[row],
                )
            else:
                unit_masks[row, unit_index, UnitAction.PASS] = True
        if orientation is Orientation.IDENTITY:
            oriented_masks = unit_masks[:, unit_index]
        else:
            # The restripe orient_unit_masks performs on a batched array, done
            # on one unit slice: oriented column j holds real column perm[j].
            oriented_masks = unit_masks[:, unit_index][:, movement_map]
        sampled_cpu, logprob, entropy = _sample_numpy_categorical(
            unit_logits[:, unit_index],
            oriented_masks,
            deterministic,
            temperature,
            generator,
        )
        # `sampled_cpu` is the oriented index the flipped forward scored; the
        # engine and every mask-evolving side effect consume its real image.
        unit_actions[:, unit_index] = sampled_cpu
        real_unit_actions[:, unit_index] = movement_map[sampled_cpu]
        unit_logprobs[:, unit_index] = logprob
        unit_entropies[:, unit_index] = entropy
        for row, raw_action in enumerate(real_unit_actions[:, unit_index]):
            if UnitAction.PLANT_WHEAT <= raw_action <= UnitAction.PLANT_MELON:
                crop = CROPS[int(raw_action) - int(UnitAction.PLANT_WHEAT)]
                remaining_seeds[row][crop] = remaining_seeds[row].get(crop, 0) - 1
            apply_unit_shed_effect(
                observations[row],
                unit_index,
                int(raw_action),
                remaining_unit_sheds[row],
                unit_tiles[row],
            )
            apply_unit_tile_effect(observations[row], unit_index, int(raw_action), unit_tiles[row])

    market_kinds = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.int64)
    market_quantities = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.int64)
    kind_masks = np.zeros((batch_size, MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.bool_)
    quantity_masks = np.zeros((batch_size, MAX_MARKET_ORDERS, N_QUANTITIES), dtype=np.bool_)
    market_active = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.bool_)
    quantity_active = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.bool_)
    kind_logprobs = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    quantity_logprobs = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    kind_entropies = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    quantity_entropies = np.zeros((batch_size, MAX_MARKET_ORDERS), dtype=np.float32)
    still_active = np.ones(batch_size, dtype=np.bool_)
    ledgers = [
        MarketLedger.from_observation(
            observation, shed=dict(remaining_unit_sheds[row]), seeds=remaining_seeds[row]
        )
        for row, observation in enumerate(observations)
    ]

    market_resources = np.zeros(
        (batch_size, MAX_MARKET_ORDERS, RESOURCE_FEATURES), dtype=np.float32
    )
    for slot in range(MAX_MARKET_ORDERS):
        resources = np.stack([market_resource_features(ledger) for ledger in ledgers])
        market_resources[:, slot] = resources
        slot_kind_logits = market_kind_logits[:, slot]
        slot_context = market_quantity_context[:, slot]
        if quantity_heads.resource_kind is not None:
            slot_kind_logits = slot_kind_logits + resources @ quantity_heads.resource_kind.T
            slot_context = slot_context + resources @ quantity_heads.resource_quantity.T
        for row, observation in enumerate(observations):
            if still_active[row]:
                kind_masks[row, slot] = _ledger_kind_mask(observation, ledgers[row])
                market_active[row, slot] = True
            else:
                kind_masks[row, slot, MarketKind.STOP] = True
        sampled_cpu, logprob, entropy = _sample_numpy_categorical(
            slot_kind_logits,
            kind_masks[:, slot],
            deterministic,
            temperature,
            generator,
        )
        market_kinds[:, slot] = sampled_cpu
        kind_logprobs[:, slot] = logprob
        kind_entropies[:, slot] = entropy

        for row, (observation, raw_kind) in enumerate(zip(observations, sampled_cpu, strict=True)):
            kind = MarketKind(int(raw_kind))
            if not still_active[row] or kind == MarketKind.STOP:
                quantity_masks[row, slot, 0] = True
                still_active[row] = False
                continue
            quantity_masks[row, slot] = _ledger_quantity_mask(observation, kind, ledgers[row])
            quantity_active[row, slot] = kind in QUANTIFIED_MARKET_KINDS
        sampled_quantity_cpu = np.zeros(batch_size, dtype=np.int64)
        logprob = np.zeros(batch_size, dtype=np.float32)
        entropy = np.zeros(batch_size, dtype=np.float32)
        active_rows = np.flatnonzero(quantity_active[:, slot])
        # Preserve the RNG stream and each row's draw while avoiding the exact
        # quantity GEMM for STOP, HIRE, BUY_LAND, and already-stopped rows. At
        # the sparse initialization policy this skips nearly all B*10*R*100
        # CPU work without changing sampled behavior.
        quantity_draws = None if deterministic else generator.random((batch_size, 1))[active_rows]
        if active_rows.size:
            active_kinds = sampled_cpu[active_rows]
            quantity_features = slot_context[active_rows] * (1.0 + quantity_kind_gate[active_kinds])
            slot_quantity_logits = (
                quantity_features @ quantity_values.T + quantity_bias[active_kinds]
            )
            if quantity_values.shape[0] == 7:
                slot_quantity_logits = percentage_quantity_logits_numpy(
                    slot_quantity_logits, quantity_masks[active_rows, slot]
                )
            elif quantity_values.shape[0] == N_QUANTITIES + 1:
                active_mask = quantity_masks[active_rows, slot]
                maximum = active_mask.sum(axis=-1).astype(np.int64) - 1
                if np.any(maximum < 0):
                    raise ValueError("quantified market rows need a legal quantity")
                rows = np.arange(active_rows.size)
                slot_quantity_logits[rows, maximum] = np.logaddexp(
                    slot_quantity_logits[rows, maximum],
                    slot_quantity_logits[:, N_QUANTITIES],
                )
                slot_quantity_logits = slot_quantity_logits[:, :N_QUANTITIES]
            active_quantities, active_logprobs, active_entropies = _sample_numpy_categorical(
                slot_quantity_logits,
                quantity_masks[active_rows, slot],
                deterministic,
                temperature,
                generator,
                draws=quantity_draws,
            )
            sampled_quantity_cpu[active_rows] = active_quantities
            logprob[active_rows] = active_logprobs
            entropy[active_rows] = active_entropies
        market_quantities[:, slot] = sampled_quantity_cpu
        quantity_logprobs[:, slot] = logprob
        quantity_entropies[:, slot] = entropy
        for row, (raw_kind, raw_quantity) in enumerate(
            zip(sampled_cpu, sampled_quantity_cpu, strict=True)
        ):
            if not market_active[row, slot] or raw_kind == MarketKind.STOP:
                continue
            _apply_ledger_order(
                observations[row],
                MarketKind(int(raw_kind)),
                QUANTITY_BINS[int(raw_quantity)],
                ledgers[row],
            )

    actions = [
        compile_action(observation, units, kinds, quantities)
        for observation, units, kinds, quantities in zip(
            observations,
            real_unit_actions,
            market_kinds,
            market_quantities,
            strict=True,
        )
    ]
    return actions
