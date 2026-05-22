import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2

from env.map_builder import MapBuilder
from env.planner import Planner


class DeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ROAD = 0
    COMPOUND = 1
    RESTAURANT = 2
    OBSTACLE = 3
    AGENT = 4
    TARGET = 5

    ACTIONS = {
        0: (-1, 0),   # UP
        1: (1, 0),    # DOWN
        2: (0, -1),   # LEFT
        3: (0, 1),    # RIGHT
    }

    def __init__(self):
        super().__init__()

        self.builder = MapBuilder()
        self.base_map = self.builder.get_base_map()
        self.compounds = self.builder.get_compounds()
        self.restaurant = self.builder.get_restaurant()

        self.height = self.base_map.shape[0]
        self.width = self.base_map.shape[1]

        self.orders_per_episode = 4
        self.max_steps = 600
        self.num_obstacles = 8

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4, self.height, self.width),
            dtype=np.float32
        )

        self.agent_pos = None
        self.obstacles = set()
        self.active_orders = []
        self.current_target = None
        self.steps = 0
        self.planner = Planner(self.base_map, self.compounds)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = self.restaurant
        self.steps = 0

        self.active_orders = random.sample(
            list(self.compounds.keys()),
            self.orders_per_episode
        )

        self.obstacles = self._generate_obstacles()

        order_id, target = self.planner.get_next_target(
            self.agent_pos,
            self.active_orders,
            self.obstacles
        )

        self.current_target = target

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        reward = -0.05
        terminated = False
        truncated = False

        self.steps += 1

        dr, dc = self.ACTIONS[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc
        new_pos = (nr, nc)

        # Out of bounds
        if not self._in_bounds(new_pos):
            reward -= 5

        # Obstacle hit
        elif new_pos in self.obstacles:
            reward -= 10

        # Compound collision
        elif self.base_map[new_pos] == self.COMPOUND:
            reward -= 5

        else:
            old_distance = self._manhattan(self.agent_pos, self.current_target)
            new_distance = self._manhattan(new_pos, self.current_target)

            self.agent_pos = new_pos

            if new_distance < old_distance:
                reward += 1
            elif new_distance > old_distance:
                reward -= 1

            # Delivery
            if self.agent_pos == self.current_target:
                reward += 100

                delivered_order = None

                for order_id in self.active_orders:
                    if self.compounds[order_id]["entrance"] == self.current_target:
                        delivered_order = order_id
                        break

                if delivered_order is not None:
                    self.active_orders.remove(delivered_order)

                if len(self.active_orders) == 0:
                    reward += 200
                    terminated = True
                else:
                    _, next_target = self.planner.get_next_target(
                        self.agent_pos,
                        self.active_orders,
                        self.obstacles
                    )

                    self.current_target = next_target

        if self.steps >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _generate_obstacles(self):
        """
        Generate safe obstacle layouts only.
        Every delivery target must remain reachable.
        """
        max_attempts = 100

        for _ in range(max_attempts):
            obstacles = set()

            entrances = {
                self.compounds[order_id]["entrance"]
                for order_id in self.active_orders
            }

            while len(obstacles) < self.num_obstacles:
                r = random.randint(0, self.height - 1)
                c = random.randint(0, self.width - 1)
                pos = (r, c)

                # Restaurant cannot be blocked
                if pos == self.restaurant:
                    continue

                # Only place obstacles on roads
                if self.base_map[pos] != self.ROAD:
                    continue

                # Delivery entrances must remain free
                if pos in entrances:
                    continue

                if pos in obstacles:
                    continue

                obstacles.add(pos)

            # Validate connectivity
            if self.planner.all_targets_reachable(
                self.restaurant,
                self.active_orders,
                obstacles
            ):
                return obstacles

        # fallback
        print("Warning: Safe obstacle generation failed. Using empty obstacle set.")
        return set()

    def _get_observation(self):
        static_channel = (self.base_map == self.COMPOUND).astype(np.float32)

        obstacle_channel = np.zeros((self.height, self.width), dtype=np.float32)
        for pos in self.obstacles:
            obstacle_channel[pos] = 1.0

        agent_channel = np.zeros((self.height, self.width), dtype=np.float32)
        agent_channel[self.agent_pos] = 1.0

        target_channel = np.zeros((self.height, self.width), dtype=np.float32)
        target_channel[self.current_target] = 1.0

        return np.stack([
            static_channel,
            obstacle_channel,
            agent_channel,
            target_channel
        ])

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def render(self):
        cell_size = 40

        canvas = np.ones(
            (self.height * cell_size, self.width * cell_size, 3),
            dtype=np.uint8
        ) * 255

        COLORS = {
            "road": (255, 255, 255),
            "compound": (120, 120, 120),
            "restaurant": (255, 0, 0),
            "obstacle": (0, 0, 255),
            "agent": (0, 255, 255),
            "target": (0, 255, 0),
            "grid": (200, 200, 200),
        }

        for r in range(self.height):
            for c in range(self.width):
                y1 = r * cell_size
                y2 = (r + 1) * cell_size
                x1 = c * cell_size
                x2 = (c + 1) * cell_size

                color = COLORS["road"]

                if self.base_map[r, c] == self.COMPOUND:
                    color = COLORS["compound"]

                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    color,
                    -1
                )

                cv2.rectangle(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    COLORS["grid"],
                    1
                )

        # obstacles
        for pos in self.obstacles:
            r, c = pos
            y1 = r * cell_size
            y2 = (r + 1) * cell_size
            x1 = c * cell_size
            x2 = (c + 1) * cell_size

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                COLORS["obstacle"],
                -1
            )

        # restaurant
        rr, rc = self.restaurant
        cv2.circle(
            canvas,
            (
                rc * cell_size + cell_size // 2,
                rr * cell_size + cell_size // 2
            ),
            cell_size // 3,
            COLORS["restaurant"],
            -1
        )

        # target
        if self.current_target is not None:
            tr, tc = self.current_target
            cv2.circle(
                canvas,
                (
                    tc * cell_size + cell_size // 2,
                    tr * cell_size + cell_size // 2
                ),
                cell_size // 3,
                COLORS["target"],
                -1
            )

        # agent
        ar, ac = self.agent_pos
        cv2.circle(
            canvas,
            (
                ac * cell_size + cell_size // 2,
                ar * cell_size + cell_size // 2
            ),
            cell_size // 3,
            COLORS["agent"],
            -1
        )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)