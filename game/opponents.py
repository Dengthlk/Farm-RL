from functools import lru_cache
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
OPPONENTS = {
    "starter": "Starter · 官方入门",
    "bc": "BC · 行为克隆",
    "ppo": "PPO · 第 1200 轮",
}


@lru_cache(maxsize=2)
def model_agent(name):
    import torch
    from kaggriculture.lejepa_model import LejepaActor, LejepaConfig
    from kaggriculture.policy import choose_actions, prepare_quantity_heads

    torch.set_num_threads(1)
    artifact = torch.load(WEIGHTS / f"{name}.pt", map_location="cpu", weights_only=True)
    actor = LejepaActor(LejepaConfig(**artifact["model_config"]))
    actor.load_state_dict(artifact["actor"])
    actor.eval()
    heads = prepare_quantity_heads(actor)

    def play(observation):
        return choose_actions(actor, [observation], heads)[0]

    return play

