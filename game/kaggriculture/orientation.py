from __future__ import annotations

import enum
import numpy as np
from kaggriculture.actions import N_UNIT_ACTIONS, UnitAction


class Orientation(enum.IntEnum):
    """One grid symmetry, named by what it does to the real world on screen.

    MIRROR_X flips the x axis (columns): the viewed board shows the real
    board's columns reversed, so the viewed EAST direction executes as real
    WEST. MIRROR_Y flips rows, so viewed NORTH executes as real SOUTH.
    ROTATE_180 applies both flips.
    """

    IDENTITY = 0
    MIRROR_X = 1
    MIRROR_Y = 2
    ROTATE_180 = 3


def movement_permutation(orientation: Orientation) -> np.ndarray:
    """Oriented action index -> real action index, identity off movement.

    Column j of a real-space mask describes real action j. Placing
    ``oriented_logits[perm]`` into real-indexed columns puts each oriented
    logit under exactly the real mask column of the action it executes, so
    sampling against real masks yields real actions directly.
    """
    permutation = np.arange(N_UNIT_ACTIONS, dtype=np.int64)
    swaps: tuple[tuple[UnitAction, UnitAction], ...] = ()
    if orientation in (Orientation.MIRROR_X, Orientation.ROTATE_180):
        swaps += ((UnitAction.EAST, UnitAction.WEST),)
    if orientation in (Orientation.MIRROR_Y, Orientation.ROTATE_180):
        swaps += ((UnitAction.NORTH, UnitAction.SOUTH),)
    for left, right in swaps:
        permutation[left], permutation[right] = permutation[right], permutation[left]
    return permutation
