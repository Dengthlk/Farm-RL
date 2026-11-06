"""Two seats and a turn counter around Kaggle's 1.32.7 game interpreter."""
from .rules import agents, interpreter, specification
from .support import Struct
from turn_motion import observe_player_actions


class Match:
    def __init__(self, seed):
        config = {
            key: value.get("default") if isinstance(value, dict) else value
            for key, value in specification["configuration"].items()
        }
        self.configuration = Struct(**dict(config, seed=seed))
        self.info = {}
        self.done = False
        self.agents = agents
        self.state = [
            Struct(observation=Struct(step=0, remainingOverageTime=60), action={}, status="ACTIVE", reward=None)
            for _ in range(2)
        ]
        self.interpreter = interpreter
        self.interpreter(self.state, self)
        self.animations = []
        observe_player_actions(self, self.animations)

    def step(self, actions):
        for seat, action in zip(self.state, actions):
            seat.action = action
        self.interpreter(self.state, self)
        for seat in self.state:
            seat.observation.step += 1
        self.done = self.state[0].status == "DONE"

