from __future__ import annotations

from dataclasses import dataclass, replace
import torch
from torch import Tensor, nn
from kaggriculture.actions import N_UNIT_ACTIONS, UnitAction
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS, PRIVATE_ITEMS
from kaggriculture.entity import EntityActor, EntityConfig, EntityTrunk
from kaggriculture.model import ActorOutput, Linear, RMSNorm
from kaggriculture.navigation import TargetNavigation, assemble_unit_logits
from kaggriculture.resource_conditioning import MarketResourceConditioner
from kaggriculture.structured import Attention, FeedForward, GatedResidual, JepaBelief, StructuredInputs
from kaggriculture.tokens import TILE_CONTINUOUS_FIELDS, TILE_COUNT


@dataclass(frozen=True)
class LejepaConfig(EntityConfig):
    """Entity geometry plus the objective's four shapes and the two towers' depth.

    The objective's knobs live on the model configuration rather than on
    `PpoConfig` because they fix parameter shapes -- the projector's hidden width
    and the direction buffers' second dimension -- and every shape a checkpoint
    has to reproduce belongs to the artifact that records it. The coefficients,
    which fix nothing, stay on `PpoConfig` where the rest of the trade-offs are.
    """

    # Schema 8 adds liquidation value, price and supply forecasts and the
    # payback of starting each crop or animal now; its leaderboard clone fits
    # and plays as the v4 one does (artifacts/probes/ppo-frontier-20260928/
    # replay). Saved artifacts still name their schema explicitly when loaded.
    observation_schema_version: int = 8
    # Every quantity choice a market kind admits (interface 2), and the local
    # unit affordance scorer below: the promoted core's configuration, which the
    # WDL clone and every PPO stage behind the default recipe trained
    # (docs/experiments/core-model-2026-09-26.md; artifacts/probes/ppo-stage2-20260927). Saved
    # artifacts from before either default load as absolute quantities without
    # the scorer (`Architecture.build_config`).
    action_interface: int = 2
    # Exact loss/draw/win classifier, the default critic head: its PPO value is
    # P(win) - P(loss), the match score the campaign selects on, and it requires
    # hard terminal outcomes with gamma and critic lambda one. HL-Gauss support
    # settings are inert in this mode; the three outcome values are always -1, 0,
    # 1. Off selects the HL-Gauss head, or the scalar one with `scalar_value`.
    wdl_value: bool = True

    #: Hidden width of every projector, of the predictor's output head, and of
    #: the reward body. `../le-wm` runs a 192-wide encoder through a 2048-wide
    #: projector hidden layer; four times the model width is the same shape of
    #: choice at this model's scale and matches the trunk's own feed-forwards.
    jepa_hidden_dim: int = 384
    #: Random directions per SIGReg evaluation. The references use 1024 over a
    #: batch of 128 single-token samples. The statistic here is taken per token
    #: position, so the population behind each one is `jepa_sigreg_rows` rather
    #: than the whole minibatch -- comparable to the reference's, and eight times
    #: it. What the direction count buys is resolution of the *sphere*, and that
    #: saturates; 128 keeps the Epps-Pulley intermediate inside a few hundred
    #: megabytes at the production minibatch and the estimator unbiased.
    jepa_slices: int = 128
    #: Farm tiles supervised per minibatch, resampled every minibatch. Predicting
    #: all two hundred at the production minibatch size would cost more than the
    #: policy update it is meant to support.
    jepa_tile_samples: int = 32
    #: Rows entering SIGReg, taken as the leading slice of an already shuffled
    #: minibatch. The statistic is a constraint on the representation, not a
    #: per-row loss, and the reference enforces it on 128 samples a step.
    jepa_sigreg_rows: int = 1024
    #: Cross-attention rounds between the belief's decision slots and their
    #: readouts, each reading the whole encoded observation. Zero makes the
    #: actor's heads the entity family's linear ones over each slot alone, which
    #: against a frozen representation is a linear probe -- the reference
    #: protocol, and worth being able to run as the control it is. It is not a
    #: usable policy: even a gated feed-forward per slot cloned to only 0.76
    #: market-kind accuracy where the end-to-end entity actor reaches 0.999,
    #: because a market slot is not where the world model keeps the economy.
    policy_readout_layers: int = 1
    # Stateless tile pointer marginalized into the four existing movement actions.
    unit_target_navigation: bool = True
    market_resource_conditioning: bool = True
    #: Let the policy's loss -- the clone loss, then PPO's -- reach the backbone
    #: beside the LeJEPA objective instead of stopping at a detach. The backbone
    #: stays owned, clipped and stepped by the objective's optimizer; it now
    #: answers to both losses. Off is the pure world-model ablation, and it does
    #: not survive sampling: its 12-epoch clone matched the teacher at argmax
    #: (0.997 market-kind accuracy) but went bankrupt in every sampled self-play
    #: game, its farm unwatered and its animals unfed within two days of the
    #: first departure, because features fitted only to predict the teacher's
    #: trajectory carry nothing a decision needs to recover off it. On, the same
    #: clone plays teacher-level banks (70-126k) under temperature-1 sampling.
    policy_shapes_backbone: bool = True
    #: Cross-attention rounds in the critic's private tower. One round is enough
    #: to let every privileged token read the whole encoded observation; the
    #: tower's job is to fold private information into a representation that is
    #: already built, not to build one.
    critic_private_layers: int = 1
    #: The entity family's critic readout feed-forward, on by default here.
    critic_readout_ffn: bool = True
    # A8: score each primitive against its destination/current tile and the
    # resource it changes. Zero-initialized query preserves a loaded A2 actor.
    unit_affordance_scorer: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.wdl_value and self.scalar_value:
            raise ValueError("the scalar value head requires wdl_value=False")
        if self.action_interface == 3:
            raise ValueError("LeJEPA decision layout does not support market-set interface 3")
        for name in ("jepa_hidden_dim", "jepa_slices", "jepa_tile_samples", "jepa_sigreg_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("policy_readout_layers", "critic_private_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.critic_private_layers < 1:
            raise ValueError(
                "the LeJEPA critic needs at least one private round; with none, its "
                "privileged inputs would reach the value head through pooling alone"
            )
        if self.jepa_tile_samples > 2 * TILE_COUNT:
            raise ValueError("jepa_tile_samples cannot exceed the encoded tile count")
        if self.critic_architecture != "entity":
            raise ValueError(
                "the LeJEPA critic reads the shared backbone; keep critic_architecture"
            )
        if self.bixt_latents:
            raise ValueError("the LeJEPA belief reads entity trunk memory, which BiXT rewrites")


class LejepaBackbone(nn.Module):
    """The shared world model: the entity trunk, read as a `JepaBelief`.

    The trunk is built with `private_columns=False` -- the actor's configuration --
    because it encodes the seat's own observation and only that. The critic's
    private columns are not missing from the model; they are embedded by the
    critic, downstream of here, which is the only arrangement in which one encoder
    can serve both seats without leaking.
    """

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = EntityTrunk(config, private_columns=False)

    def _encode(
        self, inputs: StructuredInputs, *, rematerialize_farms: bool = False
    ) -> tuple[Tensor, Tensor]:
        """States and source memory from one trunk pass, with no optional output.

        `EntityTrunk.forward_with_memory` also returns a validity mask that is
        `None` without opponent-unit memory, which this encoder never has.
        """
        states, memory, _ = self.trunk.forward_with_memory(
            inputs, rematerialize_farms=rematerialize_farms
        )
        return states, memory

    def _belief(self, states: Tensor, memory: Tensor) -> JepaBelief:
        units, markets = states.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
        tiles, economy = memory.split((2 * TILE_COUNT, memory.shape[1] - 2 * TILE_COUNT), dim=1)
        if economy.shape[1] < 1:
            raise ValueError("backbone memory carries no economy tokens")
        return JepaBelief(units, markets, economy, tiles)

    def forward(self, inputs: StructuredInputs) -> JepaBelief:
        return self._belief(*self._encode(inputs))

    def rematerialized(self, inputs: StructuredInputs) -> JepaBelief:
        """`forward`, whose trunk replays its farm blocks in backward when training.

        The belief carries the trunk's whole tile memory and the objective scores
        a sample of its columns, so the update retains the trunk's activations
        across every term that reads it. The farm blocks, which run over every
        tile token, are most of those, and only they are recomputed. At the
        production minibatch that peaks ~3.7 GiB under recomputing the whole
        trunk, as this once did, and skips the rest of its replay.

        Only this path pays the replay. `forward` keeps the farm activations:
        the clone and the critic fit without recomputing, and recomputing there
        more than doubled a clone epoch (11 s to 26 s).
        """
        return self._belief(*self._encode(inputs, rematerialize_farms=True))


class _ReadoutRound(nn.Module):
    """One round of a tower: its tokens read the world, then each other.

    Cross-attention first, and deliberately. The critic's private tokens arrive as
    bare embeddings of privileged fields, and the actor's decision slots hold what
    the world model chose to keep there; the detached latents are the
    representation it has already built. Letting a tower's tokens talk among
    themselves before they have read the observation would be a round spent on the
    thinner half.
    """

    def __init__(self, config: EntityConfig) -> None:
        super().__init__()
        self.cross_norm = RMSNorm(config.model_dim)
        self.cross_attention = Attention(config)
        self.cross_gate = GatedResidual(config.model_dim)
        self.self_norm = RMSNorm(config.model_dim)
        self.self_attention = Attention(config)
        self.self_gate = GatedResidual(config.model_dim)
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = FeedForward(config)
        self.ffn_gate = GatedResidual(config.model_dim)

    def forward(
        self, tokens: Tensor, context: Tensor, tokens_valid: Tensor, context_valid: Tensor
    ) -> Tensor:
        valid = tokens_valid.unsqueeze(-1)
        query = self.cross_norm(tokens)
        read = self.cross_attention(query, context, context_valid=context_valid)
        tokens = torch.where(valid, self.cross_gate(tokens, read), 0.0)
        query = self.self_norm(tokens)
        tokens = torch.where(
            valid,
            self.self_gate(tokens, self.self_attention(query, query, context_valid=tokens_valid)),
            0.0,
        )
        return torch.where(valid, self.ffn_gate(tokens, self.ffn(self.ffn_norm(tokens))), 0.0)


class _UnitAffordanceScorer(nn.Module):
    """A low-rank local-effect residual over the existing primitive logits.

    It reads only the acting seat's observation. Masks and sequential shed/tile
    effects stay with the native selector; this scorer never claims an action is
    legal. The unit query is zero at initialization, preserving A2 exactly.
    """

    def __init__(self, model_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.query = Linear(model_dim, rank, bias=False)
        nn.init.zeros_(self.query.weight)
        self.tile_kind = nn.Embedding(6, rank)
        self.tile_occupant = nn.Embedding(9, rank)
        self.tile_continuous = Linear(len(TILE_CONTINUOUS_FIELDS), rank, bias=False)
        self.action = nn.Embedding(N_UNIT_ACTIONS, rank)
        self.resource = Linear(5, rank, bias=False)
        self.rank = rank

        item = [0] * N_UNIT_ACTIONS
        has_item = [False] * N_UNIT_ACTIONS
        amount = [0.0] * N_UNIT_ACTIONS
        for first, count, item_index in (
            (6, 16, 0),
            (22, 8, 8),
            (30, 4, 9),
            (34, 4, 10),
            (38, 4, 11),
        ):
            for quantity in range(count):
                action_id = first + quantity
                item[action_id] = item_index
                has_item[action_id] = True
                amount[action_id] = (quantity + 1) / 16.0
        for action_id, item_index in ((42, 9), (43, 10), (44, 11)):
            item[action_id] = item_index
            has_item[action_id] = True
            amount[action_id] = 1.0 / 16.0
        for item_index in range(9):
            action_id = 59 + item_index
            item[action_id] = item_index
            has_item[action_id] = True
        self.register_buffer("item", torch.tensor(item), persistent=False)
        self.register_buffer("has_item", torch.tensor(has_item), persistent=False)
        self.register_buffer("amount", torch.tensor(amount), persistent=False)
        self.register_buffer(
            "crop",
            torch.tensor([min(4, max(0, a - 45)) for a in range(N_UNIT_ACTIONS)]),
            persistent=False,
        )
        self.register_buffer(
            "is_plant",
            torch.tensor([45 <= a <= 49 for a in range(N_UNIT_ACTIONS)]),
            persistent=False,
        )

    def forward(self, units: Tensor, inputs: StructuredInputs) -> Tensor:
        _, count, _ = units.shape
        # Own-farm tokens lead the two-farm observation. Only local verbs get
        # a residual; movement remains under the original policy head.
        own_kind = inputs.tile_categorical[:, :TILE_COUNT, 0]
        own_occupant = inputs.tile_categorical[:, :TILE_COUNT, 1]
        own_features = inputs.tile_continuous[:, :TILE_COUNT].float()
        tiles = (
            self.tile_kind(own_kind)
            + self.tile_occupant(own_occupant)
            + self.tile_continuous(own_features)
        )
        here = tiles.gather(1, inputs.unit_tile_gather[..., 0, None].expand(-1, -1, self.rank))
        here = torch.where(inputs.unit_tile_gather_valid[..., 0, None], here, 0.0)

        # Product and animal shed stock are observable private fields; seeds
        # are a separate crop family. These are turn-start amounts. Subsequent
        # selected actions change the selector ledger and its legality masks.
        held = inputs.unit_continuous[..., : len(PRIVATE_ITEMS)].index_select(2, self.item)
        held = held * self.has_item.to(held.dtype)
        shed = torch.cat((inputs.products[..., 3], inputs.animals[..., 1]), dim=1)
        shed = shed.index_select(1, self.item)[:, None, :].expand(-1, count, -1)
        shed = shed * self.has_item.to(shed.dtype)
        seed = inputs.crops[..., 1].index_select(1, self.crop)[:, None, :]
        seed = seed.expand(-1, count, -1) * self.is_plant.to(seed.dtype)
        query = self.query(units)
        tile_score = (query * here).sum(dim=-1, keepdim=True)
        action_score = torch.einsum("bur,ar->bua", query, self.action.weight)
        coefficients = torch.nn.functional.linear(query, self.resource.weight.T)
        resource_score = (
            coefficients[..., 0, None] * held.float()
            + coefficients[..., 1, None] * shed.float()
            + coefficients[..., 2, None] * seed.float()
            + coefficients[..., 3, None] * self.amount[None, None, :]
            + coefficients[..., 4, None] * inputs.town[:, None, 3, None].float()
        )
        score = (tile_score + action_score + resource_score) * (self.rank**-0.5)
        local_verb = torch.arange(N_UNIT_ACTIONS, device=score.device) >= int(UnitAction.DROP)
        return torch.where(inputs.unit_active[..., None] & local_verb, score, 0.0)


class LejepaActor(EntityActor):
    """Entity actor whose trunk is the shared world model its policy also shapes.

    `EntityActor.__init__` builds `self.trunk`; this family replaces it with the
    backbone module, so the actor's flat state dict still carries the encoder
    under `trunk.` keys and every snapshot, bundle and portable rewrite keeps
    working unchanged.
    """

    def __init__(self, config: LejepaConfig | None = None) -> None:
        super().__init__(config or LejepaConfig())

    def _build_trunk(self, config: EntityConfig) -> nn.Module:
        return LejepaBackbone(config)

    def _initialize_heads(self, config: EntityConfig) -> None:
        super()._initialize_heads(config)
        assert isinstance(config, LejepaConfig)
        if config.unit_target_navigation:
            # Initialize using the canonical action-index priors, then retain
            # only PASS/local-operation rows. No unused movement parameters.
            head = self.unit_head[-1]
            head.weight = nn.Parameter(torch.cat((head.weight[:1], head.weight[5:])).detach())
            head.bias = nn.Parameter(torch.cat((head.bias[:1], head.bias[5:])).detach())
            head.out_features = N_UNIT_ACTIONS - 4
        layers = config.policy_readout_layers
        # A state read with no live shortcut to protect, as in the critic's tower.
        readout_config = (
            replace(config, zero_init_branches=False) if config.zero_init_branches else config
        )
        self.readout_context_norm = RMSNorm(config.model_dim) if layers else None
        self.readout = nn.ModuleList(_ReadoutRound(readout_config) for _ in range(layers))
        # The rounds end on a gated residual sum, which is not unit-RMS;
        # `initialize_policy_heads` scales every readout below for one that is.
        self.unit_readout_norm = RMSNorm(config.model_dim) if layers else None
        self.market_readout_norm = RMSNorm(config.model_dim) if layers else None
        self.unit_affordance = (
            _UnitAffordanceScorer(config.model_dim) if config.unit_affordance_scorer else None
        )
        self.navigation = (
            TargetNavigation(config.model_dim) if config.unit_target_navigation else None
        )
        self.market_resource_conditioner = (
            MarketResourceConditioner(config.quantity_rank)
            if config.market_resource_conditioning
            else None
        )

    def backbone_parameters(self):
        """The world model's parameters, which the actor's optimizer must not own."""
        return self.trunk.parameters()

    def head_parameters(self):
        """Everything the policy objective may train: the actor minus its backbone."""
        backbone = {id(parameter) for parameter in self.trunk.parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in backbone)

    def encode_belief(self, inputs: StructuredInputs) -> JepaBelief:
        return self.trunk(inputs)

    def auxiliary_belief(
        self, inputs: StructuredInputs, *, rematerialize: bool = True
    ) -> JepaBelief:
        """The belief the objective reads; `rematerialize` replays the farm blocks."""
        return self.trunk.rematerialized(inputs) if rematerialize else self.trunk(inputs)

    def forward_with_belief(
        self, inputs: StructuredInputs, market_resources: Tensor | None = None
    ) -> tuple[ActorOutput, JepaBelief]:
        belief = self.encode_belief(inputs)
        return self.decode_belief(belief, inputs.unit_active, inputs, market_resources), belief

    def forward_with_auxiliary_belief(
        self,
        inputs: StructuredInputs,
        market_resources: Tensor | None = None,
        *,
        rematerialize: bool = True,
    ) -> tuple[ActorOutput, JepaBelief]:
        belief = self.auxiliary_belief(inputs, rematerialize=rematerialize)
        return self.decode_belief(belief, inputs.unit_active, inputs, market_resources), belief

    def forward(
        self, inputs: StructuredInputs, market_resources: Tensor | None = None
    ) -> ActorOutput:
        return self.forward_with_belief(inputs, market_resources)[0]

    def _read(
        self, units: Tensor, markets: Tensor, belief: JepaBelief, unit_active: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The decision slots after the readout rounds, renormalized for the heads.

        Inactive unit slots are neither read nor attended to, the trunk's own
        convention; the belief already holds them at zero.
        """
        assert self.readout_context_norm is not None
        assert self.unit_readout_norm is not None and self.market_readout_norm is not None
        batch = units.shape[0]
        always = torch.ones(
            batch,
            MAX_MARKET_ORDERS + belief.economy.shape[1] + belief.tiles.shape[1],
            dtype=torch.bool,
            device=units.device,
        )
        context = self.readout_context_norm(torch.cat(tuple(belief), dim=1))
        context_valid = torch.cat((unit_active, always), dim=1)
        decisions = torch.cat((units, markets), dim=1)
        decisions_valid = context_valid[:, : MAX_UNITS + MAX_MARKET_ORDERS]
        for round_ in self.readout:
            decisions = round_(decisions, context, decisions_valid, context_valid)
        units, markets = decisions.split((MAX_UNITS, MAX_MARKET_ORDERS), dim=1)
        return self.unit_readout_norm(units), self.market_readout_norm(markets)

    def decode_belief(
        self,
        belief: JepaBelief,
        unit_active: Tensor,
        inputs: StructuredInputs | None = None,
        market_resources: Tensor | None = None,
    ) -> ActorOutput:
        """Read the belief through the heads, attached unless the config detaches.

        This is the single point that decides whether the policy's gradient
        reaches the world model. It sits here rather than in the trainer so that
        every consumer -- PPO, behavior cloning, a rollout, an inference bundle --
        gets the same model, and no caller can change the family by forgetting to
        detach or by detaching. Under `no_grad` it costs nothing.

        The head normalizations run here rather than at encode time, which is
        where `EntityActor` puts them. They cannot run there in this family:
        the belief is what the world model predicts, and it has to be the
        encoder's own output, not a rescaling of it that the policy owns. They
        still have to run, because `initialize_policy_heads` scales every
        readout below for a unit-RMS input.
        """
        if not self.config.policy_shapes_backbone:
            belief = belief.detach()
        units = self.unit_head[0](belief.unit_decisions)
        markets = self.market_norm(belief.market_decisions)
        if self.readout:
            units, markets = self._read(units, markets, belief, unit_active)
        unit_logits = self.unit_head[-1](units)
        if self.unit_affordance is not None:
            if inputs is None:
                raise ValueError("unit affordance scorer needs the policy observation")
            affordance = self.unit_affordance(units, inputs).to(unit_logits.dtype)
            if self.navigation is not None:
                affordance = torch.cat((affordance[..., :1], affordance[..., 5:]), dim=-1)
            unit_logits = unit_logits + affordance
        if self.navigation is not None:
            if inputs is None:
                raise ValueError("target navigation needs unit positions")
            unit_logits = assemble_unit_logits(
                unit_logits, self.navigation(units, belief.tiles, inputs.unit_categorical)
            )
        output = ActorOutput(
            unit_logits=unit_logits.contiguous(),
            market_kind_logits=self.market_kind(markets).contiguous(),
            market_quantity_context=self.market_quantity_context(markets).contiguous(),
        )
        if market_resources is not None and self.market_resource_conditioner is not None:
            kinds, quantities = self.market_resource_conditioner.condition(
                output.market_kind_logits, output.market_quantity_context, market_resources
            )
            output = output._replace(market_kind_logits=kinds, market_quantity_context=quantities)
        return output
