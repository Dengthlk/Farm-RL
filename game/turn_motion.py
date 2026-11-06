"""Observe completed unit operations in the pinned engine, before market and day reset.

Each game gets its own interpreter namespace. Calls still execute the original
unit function exactly once. No module globals or shared game state are patched.
"""
from types import FunctionType


def observe_player_actions(env, receipts):
    interpreter = env.interpreter
    execute = interpreter.__globals__["_apply_unit_action"]
    own_farm = None

    def observed(farm, private, idx, action, *settings, **kwargs):
        op = action[0] if isinstance(action, list) and action else ""
        tracked = {"NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST"}
        if farm is not own_farm or op not in tracked:
            return execute(farm, private, idx, action, *settings, **kwargs)
        position = lambda: list(farm["farmer"] if idx == 0 else farm["hands"][idx - 1])
        start = position()
        x, y = start
        cell = farm["tiles"][y][x]
        was_empty = cell is None
        was_watered = isinstance(cell, dict) and cell.get("watered_today", False)
        old_bag = dict(private["inventories"][idx]) if op == "HARVEST" else {}
        result = execute(farm, private, idx, action, *settings, **kwargs)
        end = position()
        cell = farm["tiles"][y][x]
        item, count = "", 0
        changed = start != end
        if op == "PLANT":
            changed = was_empty and isinstance(cell, dict) and cell.get("crop") == action[1]
            if changed:
                item = cell["crop"]
        elif op == "WATER":
            changed = not was_watered and isinstance(cell, dict) and cell.get("watered_today", False)
        elif op == "HARVEST":
            for product, amount in private["inventories"][idx].items():
                count = amount - old_bag.get(product, 0)
                if count > 0:
                    item = product
                    break
            changed = bool(item)
        if changed:
            receipts.append(dict(unit=idx, op=op, start=start, end=end, item=item, amount=count))
        return result

    namespace = dict(interpreter.__globals__, _apply_unit_action=observed)
    observed_interpreter = FunctionType(interpreter.__code__, namespace, interpreter.__name__,
                                        interpreter.__defaults__, interpreter.__closure__)

    def run(state, environment):
        nonlocal own_farm
        receipts.clear()
        own_farm = state[0].observation.farms[0]
        try:
            return observed_interpreter(state, environment)
        finally:
            own_farm = None

    env.interpreter = run

