import math
import random
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from tarware.astar import astar_path
from tarware.definitions import (Action, AgentType, Direction,
                                 RewardType, CollisionLayers,
                                 PackageType, PACKAGE_REQUIREMENTS,
                                 PACKAGE_REWARD_SCALE)
from tarware.spaces import observation_map
from tarware.utils import find_sections, get_next_micro_action

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5
_BATTERY_FULL = 100
_BATTERY_CONSUMPTION_MOVE = 1
_BATTERY_CONSUMPTION_LOAD = 2
_BATTERY_CHARGE_RATE = 5
_BATTERY_EXP_K = 3.0        # Steepness of exponential battery reward curve
_BATTERY_EXP_SCALE = 0.002  # Per-step reward magnitude scale
KINETIC_CONSUMPTION = _BATTERY_CONSUMPTION_MOVE
LOADED_FACTOR = _BATTERY_CONSUMPTION_LOAD / _BATTERY_CONSUMPTION_MOVE
STANDBY_CONSUMPTION = 0.1

# Slipstreaming (commensalism): energy discount when following a recently-cleared path.
_SLIPSTREAM_WINDOW   = 20   # steps within which the cell counts as "recently traversed"
_SLIPSTREAM_DISCOUNT = 0.12 # fractional energy reduction (12%)

# Heterogeneous energy model parameters
_PKG_WEIGHT_MU = 5.0       # Mean package weight (kg)
_PKG_WEIGHT_SIGMA = 2.0    # Package weight std dev (kg)
_PKG_WEIGHT_MIN = 0.5      # Minimum package weight (kg)
_PKG_WEIGHT_MAX = 20.0     # Maximum package weight (kg)
_AGV_WEIGHT_FACTOR = 0.04  # Fractional extra energy per kg when loaded moving
_PICKER_LIFT_ENERGY = 1.0  # Extra picker energy per lift assist action

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y

class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path: Optional[List[Tuple[int, int]]] = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type
        self.target = 0
        self.battery = float(random.randint(20, _BATTERY_FULL))  # New: Battery level
        self.charging = False
        self.total_lifts = 0       # Cumulative lift assists (for picker energy tracking)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir

class Shelf(Entity):
    counter = 0

    def __init__(self, x, y, package_type: PackageType = PackageType.STANDARD):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.package_type = package_type
        req_agvs, req_pickers = PACKAGE_REQUIREMENTS[package_type]
        self.required_agvs = req_agvs
        self.required_pickers = req_pickers
        self.reward_scale = PACKAGE_REWARD_SCALE[package_type]
        # AGVs (and pickers for STANDARD) committed to this task. Lead carrier is index 0.
        self.carrier_group: List["Agent"] = []
        # Set True when a picker provides optional proximity assist on a HEAVY shelf.
        # Triggers 30% energy discount in _movement_energy_cost.
        self.picker_assisted: bool = False

        # Package weight sampled from truncated normal distribution. Larger packages
        # are biased heavier to keep battery-cost realism in line with type.
        type_weight_bias = {
            PackageType.SOLO:        0.6,
            PackageType.STANDARD:    1.0,
            PackageType.LARGE:       1.6,
            PackageType.HEAVY:       1.4,   # heavy but doable alone; picker assist helps
            PackageType.PICKER_SOLO: 0.3,   # light item handled by picker
        }.get(package_type, 1.0)
        self.weight = float(np.clip(
            np.random.normal(_PKG_WEIGHT_MU * type_weight_bias, _PKG_WEIGHT_SIGMA),
            _PKG_WEIGHT_MIN, _PKG_WEIGHT_MAX
        ))

class ChargingStation(Entity):
    counter = 0

    def __init__(self, x, y):
        ChargingStation.counter += 1
        super().__init__(ChargingStation.counter, x, y)

class StuckCounter:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.count = 0

    def update(self, new_position: Tuple[int, int]):
        if new_position == self.position:
            self.count += 1
        else:
            self.count = 0
            self.position = new_position

    def reset(self, position=None):
        self.count = 0
        if position:
            self.position = position

class Warehouse(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        num_agvs: int,
        num_pickers: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        normalised_coordinates: bool=False,
        observation_type: str = "global",
        package_distribution: Optional[Dict[PackageType, float]] = None,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param num_agvs: Number of spawned and controlled agv
        :type num_agvs: int
        :param num_pickers: Number of spawned and controlled pickers
        :type num_pickers: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param observation_type: Specifies type of observations
        :type normalised_coordinates: str
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers

        self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        self._charging_station_positions: set = {(s.x, s.y) for s in self.charging_stations}
        self.shelf_action_ids = list(range(len(self.goals) + 1, len(self.action_id_to_coords_map) - len(self.charging_stations) + 1))
        # If no Pickers are generated, AGVs can perform picks independently
        if num_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(num_agvs)] + [AgentType.PICKER for _ in range(num_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.num_agents)]

        # Package-size heterogeneity: default to STANDARD-only (current behaviour).
        # Keys may be PackageType enum or int; values must be non-negative and normalise.
        self.package_distribution = self._normalise_package_distribution(package_distribution)

        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        # Initialize inactive steps counter to 0 to avoid comparisons with None before reset()
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.max_steps = max_steps

        self.action_size = len(self.action_id_to_coords_map) + 1
        self.action_space = spaces.Tuple(tuple(self.num_agents * [spaces.Discrete(self.action_size)]))

        self.observation_space_mapper = observation_map[observation_type](
            self.num_agvs,
            self.num_pickers,
            self.grid_size,
            len(self.action_id_to_coords_map)-len(self.goals),
            normalised_coordinates,
        )
        self.observation_space = spaces.Tuple(tuple(self.observation_space_mapper.ma_spaces))

        self.request_queue_size = request_queue_size
        self.request_queue = []
        self.rack_groups = find_sections(list([loc for loc in self.action_id_to_coords_map.values() if (loc[1], loc[0]) not in self.goals]))
        self.agents: List[Agent] = []
        self.stuck_counters = []
        self.renderer = None

    @staticmethod
    def _normalise_package_distribution(dist: Optional[Dict]) -> Dict[PackageType, float]:
        if dist is None:
            return {PackageType.STANDARD: 1.0}
        # Accept string keys ("SOLO"), int keys, or enum keys for YAML friendliness.
        clean: Dict[PackageType, float] = {}
        for k, v in dist.items():
            if isinstance(k, PackageType):
                key = k
            elif isinstance(k, str):
                key = PackageType[k.upper()]
            else:
                key = PackageType(int(k))
            clean[key] = float(v)
        total = sum(clean.values())
        assert total > 0, "package_distribution must have positive mass"
        return {k: v / total for k, v in clean.items()}

    def _sample_package_type(self) -> PackageType:
        types = list(self.package_distribution.keys())
        probs = np.array([self.package_distribution[t] for t in types], dtype=float)
        idx = int(np.random.choice(len(types), p=probs))
        return types[idx]

    @property
    def targets_agvs(self):
        return [agent.target for agent in self.agents[:self.num_agvs]]

    @property
    def targets_pickers(self):
        return [agent.target for agent in self.agents[self.num_agvs:]]

    def _make_layout_from_params(self, shelf_columns: int, shelf_rows: int, column_height: int) -> None:
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        self._bottom_rows = 2
        self._highway_lanes = 2
        self.column_width = 2
        self.column_height = column_height
        self.grid_size = (
            self._highway_lanes + (self.column_height + self._highway_lanes) * shelf_rows  + self._bottom_rows + 1,
            self._highway_lanes + (self.column_width  + self._highway_lanes) * shelf_columns,
        )
        self.grid = np.zeros((len(CollisionLayers), *self.grid_size), dtype=np.int32)

        def get_highway_lanes_indices(axis_size, step):
            return [
                i + j
                for i in range(
                    0, axis_size, step + self._highway_lanes
                )
                for j in range(self._highway_lanes)
            ]

        highway_ys = get_highway_lanes_indices(self.grid_size[0], self.column_height)
        highway_xs = get_highway_lanes_indices(self.grid_size[1], self.column_width)

        def highway_func(x, y):
            return x in highway_xs or y in highway_ys or y >= self.grid_size[0] - 1 - self._bottom_rows

        self.goals = [
            (i, self.grid_size[0] - 1)
            for i in range(self.grid_size[1]) if not i in highway_xs
        ]
        self.num_goals = len(self.goals)

        # Charging stations: fewer than agents to create genuine resource contention.
        # Ratio 0.65 means ~2/3 of agents can charge simultaneously, forcing queuing
        # and enabling parasitism (blocking) and competition (contention) to emerge.
        n_stations = max(2, math.ceil(self.num_agents * 0.65))
        self.charging_stations = [ChargingStation(i, 0) for i in range(self.grid_size[1])][:n_stations]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        self.action_id_to_coords_map = {i+1: (x, y) for i, (y, x) in enumerate(self.goals)}
        item_loc_index=len(self.action_id_to_coords_map)+1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.action_id_to_coords_map[item_loc_index] = (y, x)
                    item_loc_index+=1
        for charging_station in self.charging_stations:
            self.action_id_to_coords_map[item_loc_index] = (charging_station.y, charging_station.x)
            item_loc_index += 1

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def find_path(self, start, goal: Tuple[int, int], agent, care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """
        Constructs a path from start to goal using the A* algorithm contidioned on a grid that integrates both the
        environment specific (shelf) and the presence of other agents as obstacles.

        If `care_for_agents` is True, the grid is adjusted to consider other agents as obtacles. However, we avoid
        situatiosns where the paths is invalidated by agents of other types waiting to cooperate with the current.
        For Pickers, the grid is further modified to ensure they can only travel through designated highways and
        access goal locations.

        Parameters:
        - care_for_agents (bool): Whether to consider other agents in the grid.
        - agent (Agent): The agent for which the path is being calculated.
        - start (tuple): The starting coordinates (x, y) of the agent.
        - goal (tuple): The goal coordinates (x, y) for the agent.

        Returns:
        - List of tuples representing the path from start to goal, or an empty list if no path is found.
        """
        grid = np.zeros(self.grid_size)
        grid += self.grid[CollisionLayers.SHELVES]
        if care_for_agents:
            grid += self.grid[CollisionLayers.AGVS]
            grid += self.grid[CollisionLayers.PICKERS]
        # Agents should start a path regardless if some others are waiting around the target location
        grid[goal[0], goal[1]] = 0

        if agent.type == AgentType.PICKER:
            # For PICKER_SOLO targets, pickers enter the aisle column to reach the shelf.
            # For all other targets, pickers are constrained to highways.
            target_shelf_id = self.grid[CollisionLayers.SHELVES, goal[0], goal[1]]
            is_picker_solo = (
                target_shelf_id and
                self.shelfs[target_shelf_id - 1].package_type == PackageType.PICKER_SOLO
            )
            if not is_picker_solo:
                grid += (1 - self.highways)
                grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            else:
                # Allow the picker to enter only the column containing the target shelf
                # (keeps them out of other aisles while enabling PICKER_SOLO access).
                col_mask = np.ones(self.grid_size, dtype=np.int32)
                col_mask[:, goal[1]] = 0
                grid += (1 - self.highways) * col_mask
                grid[goal[0], goal[1]] = 0
            for i in range(self.grid_size[1]):
                grid[self.grid_size[0] - 1, i] = 1

        # Ban Pickers crossing through racks if adjacent target location is chosen and force them thake the long way around.
        start_fix = (0, 0)
        if agent.type == AgentType.PICKER and  ((not self._is_highway(start[1], start[0])) and goal[0] == start[0] and abs(goal[1] - start[1]) == 1):
            if self._is_highway(start[1] - 1, start[0]):
                start_fix = (0, - 1)
            if self._is_highway(start[1] + 1, start[0]):
                start_fix = (0, 1)
            grid[start[0], start[1]] = 1

        grid[start[0]+start_fix[0], start[1]+start_fix[1]] = 0
        grid = [list(map(int, l)) for l in (grid!=0)]
        grid = np.array(grid, dtype=np.float32)
        grid[np.where(grid == 1)] = np.inf
        grid[np.where(grid == 0)] = 1
        adjusted_start = (start[0] + start_fix[0], start[1] + start_fix[1])
        result = astar_path(grid, adjusted_start, goal, allow_diagonal=False) # returns None if cant find path
        if result is not None:
            path_coords = [tuple(x) for x in list(result)]
            path_coords = path_coords[1 - int(grid[start[0], start[1]] > 1):]
        else:
            path_coords = []

        if path_coords:
            return [(x, y) for y, x in path_coords]
        else:
            return []

    def _recalc_grid(self) -> None:
        self.grid.fill(0)

        carried_shelf_ids = {agent.carrying_shelf.id for agent in self.agents if agent.carrying_shelf}
        for shelf in self.shelfs:
            if shelf.id not in carried_shelf_ids:
                self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] = shelf.id
        for agent in self.agents:
            layer = CollisionLayers.PICKERS if agent.type == AgentType.PICKER else CollisionLayers.AGVS
            self.grid[layer, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                 self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = agent.carrying_shelf.id

    def get_carrying_shelf_information(self):
        return [agent.carrying_shelf != None for agent in self.agents[:self.num_agvs]]

    def get_shelf_request_information(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        request_item_map = np.zeros(len(self.shelfs), dtype=int)
        requested_shelf_ids = [shelf.id for shelf in self.request_queue]
        for id_ in self.shelf_action_ids:
            coords = self.action_id_to_coords_map[id_]
            if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] in requested_shelf_ids:
                request_item_map[id_ - len(self.goals) - 1] = 1
        return request_item_map

    def get_empty_shelf_information(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        empty_item_map = np.zeros(len(self.shelfs), dtype=int)
        for id_ in self.shelf_action_ids:
            coords = self.action_id_to_coords_map[id_]
            if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] == 0 and (
                self.grid[CollisionLayers.CARRIED_SHELVES, coords[0], coords[1]] == 0
                or self.agents[
                    self.grid[CollisionLayers.AGVS, coords[0], coords[1]] - 1
                ].req_action
                not in [Action.NOOP, Action.TOGGLE_LOAD]
            ):
                empty_item_map[id_ - len(self.goals) - 1] = 1
        return empty_item_map

    def attribute_macro_actions(self, macro_actions: List[int]) -> Tuple[int, int]:
        agvs_distance_travelled = 0
        pickers_distance_travelled = 0
        # Logic for Macro Actions
        for agent, macro_action in zip(self.agents, macro_actions):
            # Initialize action for step
            agent.req_action = Action.NOOP
            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
            if not agent.busy:
                agent.target = 0
                if macro_action != 0:
                    agent.path = self.find_path((agent.y, agent.x), self.action_id_to_coords_map[macro_action], agent, care_for_agents=False)
                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
            else:
                # Check if agent finished the given path, if not continue the path
                if agent.path is None or agent.path == []:
                    # Arrival at a charging station: charge instead of TOGGLE_LOAD.
                    if (agent.x, agent.y) in self._charging_station_positions:
                        agent.req_action = Action.CHARGE
                        agent.busy = False
                    elif agent.type in [AgentType.AGV, AgentType.AGENT]:
                        agent.req_action = Action.TOGGLE_LOAD
                    elif agent.type == AgentType.PICKER:
                        # If the picker has arrived at a PICKER_SOLO shelf, execute
                        # it solo. Otherwise become available for a new assignment.
                        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
                        if (shelf_id
                                and self.shelfs[shelf_id - 1].package_type == PackageType.PICKER_SOLO
                                and self.shelfs[shelf_id - 1] in self.request_queue):
                            agent.req_action = Action.TOGGLE_LOAD
                        else:
                            agent.busy = False
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    agvs_distance_travelled += int(agent.type == AgentType.AGV)
                    pickers_distance_travelled += int(agent.type == AgentType.PICKER)
                if agent.path is not None and len(agent.path) == 1:
                    # If agent is at the end of a path and carrying a shelf and the target location is already occupied, restart agent
                    if agent.carrying_shelf and self.grid[CollisionLayers.SHELVES, agent.path[-1][1], agent.path[-1][0]]:
                        agent.req_action = Action.NOOP
                        agent.busy = False
                    # Logic for Pickers to load shelves if AGV is present at location or wait otherwise
                    if agent.type == AgentType.PICKER:
                        agv_at_target_id = self.grid[CollisionLayers.AGVS, agent.path[-1][1], agent.path[-1][0]]
                        if (
                            agv_at_target_id == 0
                            or self.agents[agv_at_target_id - 1].req_action != Action.TOGGLE_LOAD
                        ):
                            agent.req_action = Action.NOOP
                        else:
                            # AGV is at target doing TOGGLE_LOAD.
                            # For STANDARD shelves: picker should also TOGGLE_LOAD from adjacent cell.
                            # _collect_toggle_helpers uses radius=1, so picker will be found.
                            target_shelf_id = self.grid[CollisionLayers.SHELVES, agent.path[-1][1], agent.path[-1][0]]
                            if target_shelf_id:
                                target_shelf = self.shelfs[target_shelf_id - 1]
                                if target_shelf.required_pickers > 0:
                                    agent.req_action = Action.TOGGLE_LOAD
                                    self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
                                    continue  # don't overwrite with NOOP
                            self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
                            
        return agvs_distance_travelled, pickers_distance_travelled

    def resolve_move_conflict(self, agent_list):
        commited_agents = set()
        G = nx.DiGraph()
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)
            if (target[0], target[1]) not in self.charging_stations:
                G.add_edge(start, target)
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[CollisionLayers.AGVS, start_node[1], start_node[0]]
                    # action = self.agents[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, start_node[1], start_node[0]]
                    if picker_id > 0:
                        commited_agents.add(picker_id)
                        continue
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[CollisionLayers.AGVS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, y, x]
                    if picker_id:
                        commited_agents.add(picker_id)
        clashes = 0
        for agent in agent_list:
            for other in agent_list:
                if agent.id != other.id:
                    agent_new_x, agent_new_y = agent.req_location(self.grid_size)
                    other_new_x, other_new_y = other.req_location(self.grid_size)
                    # Clash fixing logic
                    if agent.path and ((agent_new_x, agent_new_y) in [(other.x, other.y), (other_new_x, other_new_y)]):
                        # If we are in a rack and one of the agents is a picker we ignore clashses, assumed behaviour is Picker is loading
                        if not self._is_highway(agent_new_x, agent_new_y) and (agent.type == AgentType.PICKER or other.type == AgentType.PICKER) and agent.type != other.type:
                            # Allow Pickers to step over AGVs (if no other Picker at that shelf location) or AGVs to step over Pickers (if no other AGV at that shelf location)
                            if ((agent.type == AgentType.PICKER and self.grid[CollisionLayers.PICKERS, agent_new_y, agent_new_x] in [0, agent.id])
                                or (agent.type == AgentType.AGV and self.grid[CollisionLayers.AGVS, agent_new_y, agent_new_x] in [0, agent.id])):
                                commited_agents.add(agent.id)
                                continue
                        # If the agent's next action bumps it into another agent
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP # Stop the action
                            # Check if the clash is not solved naturaly by the other agent moving away
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:# If the others are not already fixing the clash
                                    clashes+=1
                                    agent.fixing_clash = _FIXING_CLASH_TIME # Agent start time for clash fixing
                                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                    if new_path != []: # If the agent can find an alternative path, assign it if not let the other solve the clash
                                        agent.path = new_path
                                    else:
                                        agent.fixing_clash = 0
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y):
                            # If the agent's next action bumps it into another agent position after they take actions simultaneously
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                agent.req_action = Action.NOOP # If the agent's actions leads them in the position of another STOP
                                agent.fixing_clash = _FIXING_CLASH_TIME  # Agent wait one step while the other moves into place

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        return clashes

    def resolve_stuck_agents(self) -> int:
        # This can happen when their goal is occupied after reaching their last step/re-calculating a path
        overall_stucks = 0
        moving_agents = [
            agent
            for agent in self.agents
            if agent.busy
            and agent.req_action not in (Action.LEFT, Action.RIGHT) # Don't count changing directions
            and (agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals) # Don't count loading or changing directions / if at goal
        ]
        for agent in moving_agents:
            agent_stuck_count = self.stuck_counters[agent.id - 1]
            agent_stuck_count.update((agent.x, agent.y))
            if _STUCK_THRESHOLD < agent_stuck_count.count < _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                agent.req_action = Action.NOOP
                if agent.path:
                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1], agent.path[-1][0]), agent)
                    # Picker should wait for AGV to arrive at destination regardless of stuck count
                    if new_path:
                        agent.path = new_path
                        if len(agent.path) == 1:
                            continue
                        agent_stuck_count.reset((agent.x, agent.y))
                        continue
                else:
                    overall_stucks += 1
                    agent.busy = False
                    agent_stuck_count.reset()
            if agent_stuck_count.count > _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                overall_stucks += 1
                agent_stuck_count.reset((agent.x, agent.y))
                agent.req_action = Action.NOOP
                agent.busy = False
        return overall_stucks

    def _execute_forward(self, agent: Agent) -> None:
        new_x, new_y = agent.req_location(self.grid_size)

        # Slipstreaming (commensalism): AGV following a path recently cleared by
        # another AGV benefits from reduced congestion / better routing information.
        # Effect is passive — the leading AGV is unaffected (commensalism +/0).
        move_cost = self._movement_energy_cost(agent)
        if agent.type in (AgentType.AGV, AgentType.AGENT):
            steps_since = self._cur_steps - self._cell_last_traversed_agv[new_y, new_x]
            if 0 < steps_since <= _SLIPSTREAM_WINDOW:
                move_cost *= (1.0 - _SLIPSTREAM_DISCOUNT)
            # Record this cell as traversed by an AGV this step.
            self._cell_last_traversed_agv[agent.y, agent.x] = self._cur_steps

        agent.x, agent.y = new_x, new_y
        agent.path = agent.path[1:]
        if agent.carrying_shelf:
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
        agent.battery = max(0, agent.battery - move_cost)

    def _movement_energy_cost(self, agent: Agent) -> float:
        if agent.type not in [AgentType.AGV, AgentType.AGENT]:
            return _BATTERY_CONSUMPTION_MOVE
        if not agent.carrying_shelf:
            return _BATTERY_CONSUMPTION_MOVE

        payload = float(max(0.0, getattr(agent.carrying_shelf, "weight", 0.0)))
        if payload == 0.0:
            base = _BATTERY_CONSUMPTION_MOVE * LOADED_FACTOR
        else:
            base = _BATTERY_CONSUMPTION_MOVE * (1.0 + _AGV_WEIGHT_FACTOR * payload)

        # HEAVY + picker-assisted: 30% energy discount (commensalism benefit to AGV).
        if (agent.carrying_shelf.package_type == PackageType.HEAVY
                and agent.carrying_shelf.picker_assisted):
            base *= 0.7
        return base

    def _execute_rotation(self, agent: Agent) -> None:
        agent.dir = agent.req_direction()

    def _collect_toggle_helpers(
        self, shelf_x: int, shelf_y: int, chebyshev_radius: int
    ) -> Tuple[List[Agent], List[Agent]]:
        """Return AGVs and pickers with ``req_action == TOGGLE_LOAD`` whose cell
        is within Chebyshev-``radius`` of (shelf_x, shelf_y). Used for LARGE
        packages where helpers stand adjacent to the shelf cell."""
        agvs: List[Agent] = []
        pickers: List[Agent] = []
        for a in self.agents:
            if a.req_action != Action.TOGGLE_LOAD:
                continue
            if max(abs(a.x - shelf_x), abs(a.y - shelf_y)) > chebyshev_radius:
                continue
            if a.type == AgentType.PICKER:
                pickers.append(a)
            else:  # AGV or AGENT
                agvs.append(a)
                # Homogeneous AGENT counts as picker too.
                if a.type == AgentType.AGENT:
                    pickers.append(a)
        return agvs, pickers

    def _collect_proximity_pickers(self, x: int, y: int, radius: int = 1) -> List[Agent]:
        """Return pickers within Chebyshev radius, regardless of their current action.
        Used for HEAVY optional assist: picker proximity is sufficient — no explicit
        TOGGLE_LOAD required. Excludes pickers already carrying a shelf."""
        return [
            a for a in self.agents
            if a.type == AgentType.PICKER
            and a.carrying_shelf is None
            and max(abs(a.x - x), abs(a.y - y)) <= radius
        ]

    def _execute_load(self, agent: Agent, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Attempt a load action at agent's cell.

        Handles four distinct cases:
          PICKER_SOLO — picker completes the task in-place; no carrying required.
          HEAVY       — AGV loads alone; any picker within radius-1 provides optional
                        proximity assist (30% energy discount en route, small picker bonus).
          STANDARD    — requires exactly 1 AGV + 1 picker simultaneously (mutualism).
          SOLO/LARGE  — AGV-only or multi-AGV (existing behaviour unchanged).
        """
        # ── PICKER_SOLO: picker arrives at shelf and processes it in place ───────
        if agent.type == AgentType.PICKER:
            shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
            if shelf_id:
                shelf = self.shelfs[shelf_id - 1]
                if (shelf.package_type == PackageType.PICKER_SOLO
                        and shelf in self.request_queue):
                    self.request_queue.remove(shelf)
                    scale = shelf.reward_scale
                    if self.reward_type == RewardType.GLOBAL:
                        rewards += scale
                    elif self.reward_type == RewardType.INDIVIDUAL:
                        rewards[agent.id - 1] += scale
                    self._picker_solo_deliveries += 1
            agent.busy = False
            return rewards

        # ── AGV-initiated loads ──────────────────────────────────────────────────
        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
        if not shelf_id:
            agent.busy = False
            return rewards
        shelf = self.shelfs[shelf_id - 1]

        # LARGE accepts helpers from adjacent cells.
        # STANDARD also uses radius=1 because the picker navigates to the shelf position
        # but lands adjacent (the AGV occupies the shelf cell — collision avoidance).
        radius = 1 if (shelf.required_agvs > 1 or shelf.required_pickers > 0) else 0
        agvs, pickers = self._collect_toggle_helpers(agent.x, agent.y, radius)

        if len(agvs) < shelf.required_agvs or len(pickers) < shelf.required_pickers:
            # Not enough helpers yet — AGV stays busy and will retry next step.
            # This allows the picker time to navigate to the shelf for STANDARD tasks.
            return rewards

        # Prefer the on-cell AGV as lead so carrier_group[0] is the shelf occupant.
        on_cell = [a for a in agvs if a.x == agent.x and a.y == agent.y]
        lead = on_cell[0] if on_cell else agvs[0]
        if lead is not agent:
            return rewards  # defer to on-cell invocation

        participating_agvs   = agvs[:shelf.required_agvs]
        participating_pickers = pickers[:shelf.required_pickers]

        # ── HEAVY: optional proximity assist (no TOGGLE_LOAD needed from picker) ─
        if shelf.package_type == PackageType.HEAVY:
            nearby_pickers = self._collect_proximity_pickers(agent.x, agent.y, radius=1)
            if nearby_pickers:
                assist_picker = nearby_pickers[0]
                shelf.picker_assisted = True
                assist_picker.battery = max(0, assist_picker.battery - _PICKER_LIFT_ENERGY * 0.5)
                assist_picker.total_lifts += 1
                if self.reward_type == RewardType.INDIVIDUAL:
                    # Small bonus rewards the picker for being in the right place.
                    rewards[assist_picker.id - 1] += 0.05 * shelf.reward_scale

        # ── Execute atomic load ──────────────────────────────────────────────────
        lead.carrying_shelf = shelf
        self.grid[CollisionLayers.SHELVES, lead.y, lead.x] = 0
        self.grid[CollisionLayers.CARRIED_SHELVES, lead.y, lead.x] = shelf_id

        # For STANDARD tasks include pickers in carrier_group so they earn delivery
        # credit — this is the mutualism incentive (both parties gain from cooperation).
        if shelf.package_type == PackageType.STANDARD:
            shelf.carrier_group = list(participating_agvs) + list(participating_pickers)
        else:
            shelf.carrier_group = list(participating_agvs)

        for a in participating_agvs:
            a.battery = max(0, a.battery - _BATTERY_CONSUMPTION_LOAD)
            a.busy = False
        for p in participating_pickers:
            if p is lead:
                continue
            p.battery = max(0, p.battery - _PICKER_LIFT_ENERGY)
            p.total_lifts += 1
            p.busy = False

        scale = shelf.reward_scale
        if self.reward_type == RewardType.GLOBAL:
            rewards += 0.5 * scale
        elif self.reward_type == RewardType.INDIVIDUAL:
            # Small load-time signal for all participants; main reward comes at delivery.
            for a in participating_agvs:
                rewards[a.id - 1] += 0.05 * scale
            for p in participating_pickers:
                rewards[p.id - 1] += 0.05 * scale
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Mid-route drop of a carried shelf. For STANDARD shelves we still
        require a picker on cell (backward-compatible). SOLO and LARGE unload
        with the lead AGV only — LARGE helpers are released after load.
        Delivery at goal cells is handled in ``process_shelf_deliveries``."""
        if (agent.x, agent.y) in self.goals or (agent.x, agent.y) in self._charging_station_positions or self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return rewards
        if self._is_highway(agent.x, agent.y) or agent.carrying_shelf is None:
            return rewards

        shelf = agent.carrying_shelf
        picker_id = self.grid[CollisionLayers.PICKERS, agent.y, agent.x]
        picker_present = bool(picker_id) and self.agents[picker_id - 1].type == AgentType.PICKER

        # STANDARD requires picker on cell as before; SOLO/LARGE don't.
        if shelf.package_type == PackageType.STANDARD and agent.type == AgentType.AGV and not picker_present:
            agent.busy = False
            return rewards

        self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = shelf.id
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        agent.carrying_shelf = None
        agent.busy = False
        agent.has_delivered = False
        agent.battery = max(0, agent.battery - _BATTERY_CONSUMPTION_LOAD)
        shelf.carrier_group = []

        if picker_present:
            picker = self.agents[picker_id - 1]
            picker.battery = max(0, picker.battery - _PICKER_LIFT_ENERGY)
            picker.total_lifts += 1

        scale = shelf.reward_scale
        if self.reward_type == RewardType.GLOBAL:
            rewards += 0.5 * scale
        elif self.reward_type == RewardType.INDIVIDUAL:
            if picker_present:
                rewards[picker_id - 1] += 0.1 * scale
            else:
                rewards[agent.id - 1] += 0.1 * scale
        return rewards

    def _execute_charge(self, agent: Agent) -> float:
        """Charge the agent and return a small reward proportional to energy gained."""
        if (agent.x, agent.y) in self._charging_station_positions:
            before = agent.battery
            agent.battery = min(_BATTERY_FULL, agent.battery + _BATTERY_CHARGE_RATE)
            return (agent.battery - before) * 0.001   # +0.005 per full charge step
        return 0.0

    def execute_micro_actions(self, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        # Charging capacity: each station serves at most one agent per step.
        # Lower agent-ID wins when multiple agents contest the same station.
        # This makes charger blocking (parasitism) and contention (competition) possible.
        #
        # Contention tracking: multiple agents targeting the same charger = contested.
        # Use agent.target (macro destination) to detect before physical arrival.
        charger_target_map: Dict[Tuple, List[int]] = {}
        charger_action_positions: set = {
            self.action_id_to_coords_map[aid]
            for aid in self.action_id_to_coords_map
            if self.action_id_to_coords_map[aid] in {(s.y, s.x) for s in self.charging_stations}
        }
        for agent in self.agents:
            if agent.target and agent.target in self.action_id_to_coords_map:
                dest = self.action_id_to_coords_map[agent.target]
                if dest in charger_action_positions:
                    charger_target_map.setdefault(dest, []).append(agent.id - 1)

        # Give charge only to the first (lowest-ID) requester per station.
        occupied_chargers: set = set()
        for agent in sorted(self.agents, key=lambda a: a.id):
            if agent.req_action == Action.CHARGE:
                pos = (agent.x, agent.y)
                if pos in self._charging_station_positions and pos not in occupied_chargers:
                    occupied_chargers.add(pos)
                    rewards[agent.id - 1] += self._execute_charge(agent)
                # else: station contested — agent gets no charge this step

        # Store contested pairs for _build_info (pairs of agents heading to same charger).
        self._contested_charger_pairs: List[Tuple[int, int]] = []
        for requesters in charger_target_map.values():
            if len(requesters) > 1:
                for i in range(len(requesters)):
                    for j in range(i + 1, len(requesters)):
                        self._contested_charger_pairs.append((requesters[i], requesters[j]))

        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_shelf:
                    rewards = self._execute_load(agent, rewards)
                else:
                    rewards = self._execute_unload(agent, rewards)
            # CHARGE already handled above; skip to avoid double processing.
        return rewards

    def process_shelf_deliveries(self, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> Tuple[np.ndarray[Any, np.dtype[np.float64]], int]:
        shelf_deliveries = 0
        deliveries_by_type = {t: 0 for t in PackageType}
        # Note: do NOT reset _picker_solo_deliveries here — it was already
        # incremented by _execute_load (called before this function). Reset after reading.
        self._delivered_this_step: List[List[Agent]] = []  # carrier groups for this step only
        for y, x in self.goals:
            shelf_id = self.grid[CollisionLayers.CARRIED_SHELVES, x, y]
            if not shelf_id or self.shelfs[shelf_id - 1] not in self.request_queue:
                continue
            shelf = self.shelfs[shelf_id - 1]
            # Remove shelf from request queue (do not add replacement)
            self.request_queue.remove(shelf)

            agent = self.agents[self.grid[CollisionLayers.AGVS, x, y] - 1]
            if not agent.has_delivered:
                agent.has_delivered = True
                scale = shelf.reward_scale
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 1.0 * scale
                elif self.reward_type == RewardType.INDIVIDUAL:
                    group = shelf.carrier_group if shelf.carrier_group else [agent]
                    share = (1.0 * scale) / len(group)
                    for a in group:
                        rewards[a.id - 1] += share
                # Record carrier group for this step's info, then clear it so it
                # doesn't persist into future steps (would create false positives in
                # the relationship classifier's picker_in_agv_group check).
                self._delivered_this_step.append(list(shelf.carrier_group))
                shelf.carrier_group = []
            shelf_deliveries += 1
            deliveries_by_type[shelf.package_type] = deliveries_by_type.get(shelf.package_type, 0) + 1
        # PICKER_SOLO deliveries happen in _execute_load; count them here.
        deliveries_by_type[PackageType.PICKER_SOLO] = self._picker_solo_deliveries
        shelf_deliveries += self._picker_solo_deliveries
        self._picker_solo_deliveries = 0  # reset after reading, not before
        self._last_deliveries_by_type = deliveries_by_type

        if shelf_deliveries:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1

        return rewards, shelf_deliveries

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        # Reset counters
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self._last_deliveries_by_type = {t: 0 for t in PackageType}
        self._picker_solo_deliveries = 0
        # Slipstream traversal timestamps: -window so no discount fires on step 0.
        self._cell_last_traversed_agv = np.full(self.grid_size, -_SLIPSTREAM_WINDOW - 1, dtype=np.int32)

        # Set seed
        self.seed(seed)

        # Make the shelfs; sample a package type per shelf from self.package_distribution.
        self.shelfs = [
            Shelf(x, y, package_type=self._sample_package_type())
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y) and (x, y) not in [(s.x, s.y) for s in self.charging_stations]
        ]
        self._higway_locs = np.array([(y, x) for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            ) if self._is_highway(x, y)])

        # Spawn agents on higwahy locations
        agent_loc_ids = np.random.choice(
            np.arange(len(self._higway_locs)),
            size=self.num_agents,
            replace=False,
        )
        agent_locs = [self._higway_locs[agent_loc_ids, 0], self._higway_locs[agent_loc_ids, 1]]
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.num_agents)
        self.agents = [
            Agent(x, y, dir_, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]
        self._recalc_grid()

        indices = np.random.choice(len(self.shelfs), size=self.request_queue_size, replace=False)
        self.request_queue = [self.shelfs[i] for i in indices]
        self.observation_space_mapper.extract_environment_info(self)
        obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])
        return obs, {}

    def step(
        self, action
    ) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        # Attribute macro actions to agents and resolve conflicts
        agvs_distance_travelled, pickers_distance_travelled = self.attribute_macro_actions(action)
        clashes_count = self.resolve_move_conflict(self.agents)
        # Restart agents if they are stuck at the same position
        stucks_count = self.resolve_stuck_agents()

        rewards = np.zeros(self.num_agents)
        # Apply penalty for inactivity
        rewards -= 0.001
        # Exponential battery reward: smoothly negative when low, positive when high.
        # Centred at b=50 (r≈0), shaped by exp so extremes are increasingly punished/rewarded.
        # r(b) = scale * (exp(k * b/100) − exp(k * 0.5))
        # At b=0: ≈ −0.007  |  b=50: 0  |  b=100: ≈ +0.031  (with k=3, scale=0.002)
        for i, agent in enumerate(self.agents):
            b_norm = agent.battery / _BATTERY_FULL
            rewards[i] += _BATTERY_EXP_SCALE * (
                np.exp(_BATTERY_EXP_K * b_norm) - np.exp(_BATTERY_EXP_K * 0.5)
            )
        # Execute micro actions
        rewards = self.execute_micro_actions(rewards)
        # Process shelf deliveries
        rewards, shelf_deliveries = self.process_shelf_deliveries(rewards)

        self._recalc_grid()
        self._cur_steps += 1
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps) or len(self.request_queue) == 0:
            terminateds = truncateds = self.num_agents * [True]
        else:
            terminateds = truncateds =  self.num_agents * [False]

        self.observation_space_mapper.extract_environment_info(self)
        new_obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])
        info = self._build_info(
            agvs_distance_travelled,
            pickers_distance_travelled,
            clashes_count,
            stucks_count,
            shelf_deliveries,
            action,
        )
        return new_obs, list(rewards), terminateds, truncateds, info

    def _build_info(
        self,
        agvs_distance_travelled: int,
        pickers_distance_travelled: int,
        clashes_count: int,
        stucks_count:  int,
        shelf_deliveries: int,
        macro_actions: List[int],
    ) -> Dict[str, np.ndarray]:
        info = {}
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[:self.num_agvs]])
        pickers_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[self.num_agvs:]])
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        info["clashes"] = clashes_count
        info["stucks"] = stucks_count
        info["agvs_distance_travelled"] = agvs_distance_travelled
        info["pickers_distance_travelled"] = pickers_distance_travelled
        info["agvs_idle_time"] = agvs_idle_time
        info["pickers_idle_time"] = pickers_idle_time
        info["battery_levels"] = [agent.battery for agent in self.agents]
        info["actions"] = macro_actions
        info["pkg_weights"] = [
            agent.carrying_shelf.weight if agent.carrying_shelf else 0.0
            for agent in self.agents
        ]
        info["total_lifts"] = [agent.total_lifts for agent in self.agents]
        info["charging_states"] = [agent.charging for agent in self.agents]
        info["deliveries_by_pkg_type"] = {
            k.name: int(v) for k, v in getattr(self, "_last_deliveries_by_type", {}).items()
        }
        # Per-agent currently-carrying package type (None if not carrying).
        info["carrying_pkg_type"] = [
            a.carrying_shelf.package_type.name if a.carrying_shelf else None
            for a in self.agents
        ]
        # contested_chargers: agent-index pairs that requested CHARGE at the same station this step.
        # Built in execute_micro_actions via _contested_charger_pairs.
        info["contested_chargers"] = getattr(self, "_contested_charger_pairs", [])
        # Carrier groups for deliveries THIS step only (cleared after recording).
        # Wrapper uses this to determine who participated in each delivery.
        info["delivery_carrier_groups"] = [
            [a.id - 1 for a in group]
            for group in getattr(self, "_delivered_this_step", [])
        ]
        return info

    def compute_valid_action_masks(self, pickers_to_agvs=True, block_conflicting_actions=True):
        requested_items = self.get_shelf_request_information()
        empty_items = self.get_empty_shelf_information()
        carrying_shelf_info = self.get_carrying_shelf_information()
        num_goal_actions = len(self.goals)
        num_location_actions = self.action_size - (1 + num_goal_actions)
        num_shelf_actions = len(self.shelf_action_ids)

        # Expand shelf-only vectors (requested/empty) to full location action width
        # by appending charging action slots. Charging targets are always selectable.
        requested_locations = np.ones(num_location_actions, dtype=int)
        requested_locations[:num_shelf_actions] = requested_items
        empty_locations = np.ones(num_location_actions, dtype=int)
        empty_locations[:num_shelf_actions] = empty_items

        targets_agvs = [target - len(self.goals) - 1 for target in self.targets_agvs if target > len(self.goals)]
        targets_pickers = [target - len(self.goals) - 1 for target in self.targets_pickers if target > len(self.goals)]

        # Per-shelf-action-index capacity: how many AGVs / pickers may target it.
        # SOLO: 1 AGV / 0 pickers. STANDARD: 1 / 1. LARGE: 2 / 2. Charging: 1 / n_a.
        agv_capacity = np.ones(num_location_actions, dtype=int)
        picker_capacity = np.zeros(num_location_actions, dtype=int)
        # Shelves that are currently requested get capacity from their package_type.
        # Non-shelf slots (charging) keep default (AGV=1, picker=0 → never valid picker target).
        for s_action_id in self.shelf_action_ids:
            coords = self.action_id_to_coords_map[s_action_id]
            shelf_grid_id = self.grid[CollisionLayers.SHELVES, coords[0], coords[1]]
            carried_grid_id = self.grid[CollisionLayers.CARRIED_SHELVES, coords[0], coords[1]]
            shelf_id = shelf_grid_id or carried_grid_id
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]
            i = s_action_id - len(self.goals) - 1
            agv_capacity[i] = shelf.required_agvs
            picker_capacity[i] = shelf.required_pickers

        # Count agents currently targeting each slot.
        agv_target_counts = np.zeros(num_location_actions, dtype=int)
        for t in targets_agvs:
            if 0 <= t < num_location_actions:
                agv_target_counts[t] += 1
        picker_target_counts = np.zeros(num_location_actions, dtype=int)
        for t in targets_pickers:
            if 0 <= t < num_location_actions:
                picker_target_counts[t] += 1

        # Compute valid location list for AGVs
        valid_location_list_agvs = np.array([
            empty_locations if carrying_shelf else requested_locations for carrying_shelf in carrying_shelf_info
        ])
        # Compute valid location list for Pickers — follow AGV targets, but only where
        # the AGV's task actually needs a picker (excludes SOLO pickups and LARGE drop-offs).
        if pickers_to_agvs:
            valid_location_list_pickers = np.zeros(num_location_actions, dtype=int)
            for agv in self.agents[:self.num_agvs]:
                if agv.target == 0:
                    continue
                t = agv.target - len(self.goals) - 1
                if t < 0 or t >= num_location_actions:
                    continue
                if agv.carrying_shelf is not None:
                    # Drop-off target: picker only needed for STANDARD.
                    if agv.carrying_shelf.package_type == PackageType.STANDARD:
                        valid_location_list_pickers[t] = 1
                else:
                    # Pickup target: picker needed if shelf requires one.
                    if t < num_shelf_actions and picker_capacity[t] > 0:
                        valid_location_list_pickers[t] = 1
            # Independently allow pickers to target PICKER_SOLO shelves in the queue.
            # These are not tied to any AGV target — pickers discover them on their own.
            for s_action_id in self.shelf_action_ids:
                i = s_action_id - len(self.goals) - 1
                coords = self.action_id_to_coords_map[s_action_id]
                shelf_grid_id = self.grid[CollisionLayers.SHELVES, coords[0], coords[1]]
                if shelf_grid_id:
                    shelf = self.shelfs[shelf_grid_id - 1]
                    if (shelf.package_type == PackageType.PICKER_SOLO
                            and shelf in self.request_queue):
                        valid_location_list_pickers[i] = 1
        else:
            valid_location_list_pickers = (requested_locations & (picker_capacity > 0).astype(int))
            # Also expose PICKER_SOLO shelves.
            for s_action_id in self.shelf_action_ids:
                i = s_action_id - len(self.goals) - 1
                coords = self.action_id_to_coords_map[s_action_id]
                shelf_grid_id = self.grid[CollisionLayers.SHELVES, coords[0], coords[1]]
                if shelf_grid_id:
                    shelf = self.shelfs[shelf_grid_id - 1]
                    if (shelf.package_type == PackageType.PICKER_SOLO
                            and shelf in self.request_queue):
                        valid_location_list_pickers[i] = 1
        # Allow pickers to independently target charging stations (indices num_shelf_actions onward).
        valid_location_list_pickers[num_shelf_actions:] = 1

        # Mask out conflicting actions for agents of the same type
        if block_conflicting_actions:
            agv_blocked = np.where(agv_target_counts >= agv_capacity)[0]
            picker_blocked = np.where(picker_target_counts >= np.maximum(picker_capacity, 1))[0]
            valid_location_list_agvs[:, agv_blocked] = 0
            valid_location_list_pickers[picker_blocked] = 0
        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:self.num_agvs,  1 + len(self.goals):] = valid_location_list_agvs
        valid_action_masks[:self.num_agvs,  1 : 1 + len(self.goals)] = np.repeat(np.expand_dims(np.array(carrying_shelf_info), 1), len(self.goals), axis=1)
        valid_action_masks[self.num_agvs:,  1 + len(self.goals):] = valid_location_list_pickers
        valid_action_masks[self.num_agvs:,  1 : 1 + len(self.goals)] = 0
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from tarware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)