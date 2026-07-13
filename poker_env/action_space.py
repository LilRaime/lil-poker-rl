import enum
import gymnasium as gym


class PokerAction(enum.IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_MIN = 2
    RAISE_POT = 3
    ALL_IN = 4


def get_action_space() -> gym.spaces.Space:
    return gym.spaces.Discrete(5)
