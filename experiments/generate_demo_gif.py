"""
Generate a demo GIF for the warehouse environment.

The demo uses the heuristic mission scheduler from tarware/heuristic.py.
This is important because TARWARE actions are macro task-allocation targets,
not low-level motion commands. The environment handles motion planning after
each task target is assigned.
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tarware  # noqa: F401
from tarware.heuristic import heuristic_episode


CELL = 18
BORDER = 1
BG = np.array([245, 243, 236], dtype=np.uint8)
GRID = np.array([214, 210, 202], dtype=np.uint8)
SHELF = np.array([68, 73, 135], dtype=np.uint8)
REQUEST = np.array([28, 142, 134], dtype=np.uint8)
GOAL = np.array([45, 45, 45], dtype=np.uint8)
CHARGE = np.array([173, 216, 230], dtype=np.uint8)
AGV = np.array([224, 122, 37], dtype=np.uint8)
AGV_LOAD = np.array([202, 52, 51], dtype=np.uint8)
PICKER = np.array([52, 152, 219], dtype=np.uint8)
TEXT = np.array([30, 30, 30], dtype=np.uint8)


parser = argparse.ArgumentParser(
    description="Generate a heuristic task-scheduling demo GIF",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env", default="tarware-small-4agvs-2pickers-partialobs-chg-v1")
parser.add_argument("--seed", default=7, type=int)
parser.add_argument("--max_steps", default=300, type=int)
parser.add_argument("--max_inactivity_steps", default=None, type=int)
parser.add_argument("--output", default="assets/robotic_symbiosis_demo.gif")
parser.add_argument("--metadata_output", default="assets/robotic_symbiosis_demo.json")


def _draw_cell(img: np.ndarray, x: int, y: int, color: np.ndarray, rows: int) -> None:
    yy = rows - y - 1
    x0 = x * (CELL + BORDER) + BORDER
    y0 = yy * (CELL + BORDER) + BORDER
    img[y0:y0 + CELL, x0:x0 + CELL] = color


def _draw_rect(img: np.ndarray, x0: int, y0: int, w: int, h: int, color: np.ndarray) -> None:
    x0 = max(0, x0)
    y0 = max(0, y0)
    img[y0:y0 + h, x0:x0 + w] = color


def render_state(env) -> np.ndarray:
    rows, cols = env.grid_size
    height = rows * (CELL + BORDER) + BORDER
    width = cols * (CELL + BORDER) + BORDER
    img = np.tile(BG, (height, width, 1))

    for r in range(rows + 1):
        y = r * (CELL + BORDER)
        img[y:y + BORDER, :, :] = GRID
    for c in range(cols + 1):
        x = c * (CELL + BORDER)
        img[:, x:x + BORDER, :] = GRID

    for gx, gy in env.goals:
        _draw_cell(img, gx, gy, GOAL, rows)
    for station in env.charging_stations:
        _draw_cell(img, station.x, station.y, CHARGE, rows)

    requested = {s.id for s in env.request_queue}
    for shelf in env.shelfs:
        if any(agent.carrying_shelf is shelf for agent in env.agents):
            continue
        _draw_cell(img, shelf.x, shelf.y, REQUEST if shelf.id in requested else SHELF, rows)

    for agent in env.agents:
        color = AGV_LOAD if agent.carrying_shelf else AGV
        if agent.type.name == "PICKER":
            color = PICKER

        yy = rows - agent.y - 1
        x0 = agent.x * (CELL + BORDER) + BORDER + 3
        y0 = yy * (CELL + BORDER) + BORDER + 3
        _draw_rect(img, x0, y0, CELL - 6, CELL - 6, color)

        cx = x0 + (CELL - 6) // 2
        cy = y0 + (CELL - 6) // 2
        if agent.dir.name == "UP":
            _draw_rect(img, cx - 1, y0, 3, 5, TEXT)
        elif agent.dir.name == "DOWN":
            _draw_rect(img, cx - 1, y0 + CELL - 11, 3, 5, TEXT)
        elif agent.dir.name == "LEFT":
            _draw_rect(img, x0, cy - 1, 5, 3, TEXT)
        elif agent.dir.name == "RIGHT":
            _draw_rect(img, x0 + CELL - 11, cy - 1, 5, 3, TEXT)

    return img


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


class DemoRenderAdapter:
    """
    Forward reset/step to the real env but render with a lightweight state renderer.

    The heuristic planner assigns macro task targets. The warehouse env then converts
    those targets into internal motion planning and execution.
    """

    def __init__(self, env: gym.Env):
        self._env = env
        self.unwrapped = env.unwrapped

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, actions):
        return self._env.step(actions)

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return True
        return render_state(self.unwrapped)

    def close(self):
        return self._env.close()


def main() -> None:
    args = parser.parse_args()
    out = Path(args.output)
    meta_out = Path(args.metadata_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env,
        max_inactivity_steps=args.max_inactivity_steps,
        max_steps=args.max_steps,
    )
    adapter = DemoRenderAdapter(env)

    try:
        result = heuristic_episode(
            adapter,
            seed=args.seed,
            save_gif=True,
            gif_path=str(out),
        )
    finally:
        adapter.close()

    all_infos = result[0]
    total_deliveries = int(sum(info.get("shelf_deliveries", 0) for info in all_infos))
    metadata = {
        "env": args.env,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "total_deliveries": total_deliveries,
        "num_steps": len(all_infos),
        "action_semantics": "macro task-planning targets; motion is handled by the environment",
        "step_log": [
            {
                "step": i,
                "actions": _to_builtin(info.get("actions", [])),
                "shelf_deliveries": _to_builtin(info.get("shelf_deliveries", 0)),
                "battery_levels": _to_builtin(info.get("battery_levels", [])),
                "charging_states": _to_builtin(info.get("charging_states", [])),
                "vehicles_busy": _to_builtin(info.get("vehicles_busy", [])),
            }
            for i, info in enumerate(all_infos)
        ],
    }

    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved demo GIF to {out}")
    print(f"Saved demo metadata to {meta_out}")
    print(f"Deliveries in demo: {total_deliveries}")


if __name__ == "__main__":
    main()
