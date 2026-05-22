from collections import deque


class Planner:
    def __init__(self, base_map, compounds):
        self.base_map = base_map
        self.compounds = compounds
        self.height = base_map.shape[0]
        self.width = base_map.shape[1]

    def get_next_target(self, current_pos, active_orders, obstacles):
        """
        Returns the nearest reachable delivery entrance.
        """
        best_order = None
        best_target = None
        best_distance = float("inf")

        for order_id in active_orders:
            target = self.compounds[order_id]["entrance"]

            dist = self.bfs_distance(current_pos, target, obstacles)

            if dist is not None and dist < best_distance:
                best_distance = dist
                best_order = order_id
                best_target = target

        return best_order, best_target

    def bfs_distance(self, start, goal, obstacles):
        """
        Shortest path length avoiding compounds + obstacles.
        """
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            pos, dist = queue.popleft()

            if pos == goal:
                return dist

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = pos[0] + dr
                nc = pos[1] + dc
                nxt = (nr, nc)

                if not self._in_bounds(nxt):
                    continue

                if nxt in visited:
                    continue

                if nxt in obstacles:
                    continue

                if self.base_map[nxt] == 1:  # COMPOUND
                    continue

                visited.add(nxt)
                queue.append((nxt, dist + 1))

        return None

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width
    
    def all_targets_reachable(self, start, active_orders, obstacles):
        """
        Ensure every delivery target can be reached.
        """
        for order_id in active_orders:
            target = self.compounds[order_id]["entrance"]

            if self.bfs_distance(start, target, obstacles) is None:
                return False

        return True