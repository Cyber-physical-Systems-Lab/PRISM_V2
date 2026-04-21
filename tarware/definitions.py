from enum import Enum, IntEnum


class AgentType(Enum):
    AGV = 0
    PICKER = 1
    AGENT = 2

class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    TOGGLE_LOAD = 4
    CHARGE = 5

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

class CollisionLayers(IntEnum):
    AGVS = 0
    PICKERS = 1
    SHELVES = 2
    CARRIED_SHELVES = 3

class PackageType(IntEnum):
    SOLO = 0         # 1 AGV, 0 pickers — AGV operates alone
    STANDARD = 1     # 1 AGV, 1 picker  — obligate cooperation (mutualism)
    LARGE = 2        # 2 AGVs, 2 pickers — multi-agent mutualism
    HEAVY = 3        # 1 AGV, 0 pickers required; picker proximity reduces AGV energy 30%
    PICKER_SOLO = 4  # 0 AGVs, 1 picker  — picker-exclusive inventory task

PACKAGE_REQUIREMENTS = {
    PackageType.SOLO:        (1, 0),
    PackageType.STANDARD:    (1, 1),
    PackageType.LARGE:       (2, 2),
    PackageType.HEAVY:       (1, 0),
    PackageType.PICKER_SOLO: (0, 1),
}

PACKAGE_REWARD_SCALE = {
    PackageType.SOLO:        0.5,
    PackageType.STANDARD:    1.0,
    PackageType.LARGE:       2.0,
    PackageType.HEAVY:       0.75,   # higher reward reflects energy cost; picker assist makes it net-positive
    PackageType.PICKER_SOLO: 0.4,    # picker earns this independently; baseline for picker fitness
}
