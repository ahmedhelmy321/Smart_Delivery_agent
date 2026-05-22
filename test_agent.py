from collections import deque
from env.delivery_env import DeliveryEnv


def bfs_next_action(env, start, goal):
    """
    Find next action using BFS shortest path.
    """
    queue = deque([(start, [])])
    visited = {start}

    action_moves = {
        0: (-1, 0),   # UP
        1: (1, 0),    # DOWN
        2: (0, -1),   # LEFT
        3: (0, 1)     # RIGHT
    }

    while queue:
        pos, path = queue.popleft()

        if pos == goal:
            return path[0] if path else 4

        for action, (dr, dc) in action_moves.items():
            nr = pos[0] + dr
            nc = pos[1] + dc
            nxt = (nr, nc)

            if not env._in_bounds(nxt):
                continue

            if nxt in visited:
                continue

            if nxt in env.obstacles:
                continue

            if env.base_map[nxt] == env.COMPOUND:
                continue

            visited.add(nxt)
            queue.append((nxt, path + [action]))

    return 4