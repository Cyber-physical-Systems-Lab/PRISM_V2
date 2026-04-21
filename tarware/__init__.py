import itertools

import gymnasium as gym

from tarware.spaces import observation_map
from tarware.warehouse import RewardType

_obs_types = list(observation_map.keys())

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
    "extralarge": (4, 7),
}

_request_queues = {
    "tiny": 20,
    "small": 20,
    "medium": 20,
    "large": 60,
    "extralarge": 100,
}

_perms = itertools.product(_sizes.keys(), _obs_types, range(1,20), range(0, 10))

# Default package mix for `-pkgmix` env IDs (SOLO/STANDARD/LARGE only).
_DEFAULT_PKG_MIX = {"SOLO": 0.3, "STANDARD": 0.4, "LARGE": 0.3}

# Symbiosis mix: includes HEAVY (optional picker assist → commensalism) and
# PICKER_SOLO (picker-exclusive tasks → picker independence, neutralism baseline).
# Distribution designed so every relationship type can emerge:
#   MUTUALISM    ← STANDARD tasks (both earn, both required)
#   COMMENSALISM ← HEAVY tasks    (AGV earns more, picker passively assists)
#   COMPETITION  ← charger contention (fewer stations than agents)
#   PARASITISM   ← task interception / charger blocking
#   NEUTRALISM   ← SOLO + PICKER_SOLO (agents operate independently)
_SYMBIOSIS_PKG_MIX = {
    "SOLO":        0.20,
    "STANDARD":    0.30,
    "LARGE":       0.10,
    "HEAVY":       0.25,
    "PICKER_SOLO": 0.15,
}

for size, obs_type, num_agvs, num_pickers in _perms:
    # normal tasks
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "num_agvs":  num_agvs,
            "num_pickers": num_pickers,
            "request_queue_size": _request_queues[size],
            "max_inactivity_steps": 100,
            "max_steps": 5000,
            "reward_type": RewardType.INDIVIDUAL,
            "observation_type": obs_type,
        },
    )
    # Package-heterogeneous variant: three package classes (SOLO/STANDARD/LARGE).
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-pkgmix-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "num_agvs":  num_agvs,
            "num_pickers": num_pickers,
            "request_queue_size": _request_queues[size],
            "max_inactivity_steps": 100,
            "max_steps": 5000,
            "reward_type": RewardType.INDIVIDUAL,
            "observation_type": obs_type,
            "package_distribution": dict(_DEFAULT_PKG_MIX),
        },
    )
    # Full-symbiosis variant: five package classes + constrained charging.
    # Designed to enable all five ecological relationship types to emerge.
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-symbiosis-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "num_agvs":  num_agvs,
            "num_pickers": num_pickers,
            "request_queue_size": _request_queues[size],
            "max_inactivity_steps": 100,
            "max_steps": 5000,
            "reward_type": RewardType.INDIVIDUAL,
            "observation_type": obs_type,
            "package_distribution": dict(_SYMBIOSIS_PKG_MIX),
        },
    )

def full_registration():
    _perms = itertools.product(_sizes.keys(), _obs_types, _request_queues, range(1,20), range(0, 10),)
    for size, obs_type, request_queue_size, num_agvs, num_pickers in _perms:
        # normal tasks with modified column height
        gym.register(
            id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-v1",
            entry_point="tarware.warehouse:Warehouse",
            disable_env_checker=True,
            kwargs={
                "column_height": 8,
                "shelf_rows": _sizes[size][0],
                "shelf_columns": _sizes[size][1],
                "num_agvs":  num_agvs,
                "num_pickers": num_pickers,
                "sensor_range": 1,
                "request_queue_size": _request_queues[request_queue_size],
                "max_inactivity_steps": None,
                "max_steps": 10000,
                "reward_type": RewardType.INDIVIDUAL,
                "observation_type": obs_type,
            },
        )
