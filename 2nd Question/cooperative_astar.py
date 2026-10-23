import heapq
from collections import defaultdict

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _reconstruct_path(came_from, state):
    path = []
    while state is not None:
        path.append((state[0], state[1]))
        state = came_from[state]
    path.reverse()
    return path

def cooperative_astar(grid, start, goal, reservations, edge_reservations, heuristic=manhattan_distance, max_time=None,):
    
    start = tuple(int(x) for x in start)
    goal = tuple(int(x) for x in goal)
    rows, cols = grid.shape

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError(f"Start {start} is outside the grid.")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError(f"Goal {goal} is outside the grid.")
    if grid[start] == 1 or grid[goal] == 1:
        return {
            "path": None, "path_length": None, "arrival_time": None,
            "nodes_expanded": 0, "max_frontier_size": 0, "success": False
        }

    if max_time is None:
        max_time = max(4 * rows * cols, manhattan_distance(start, goal) + rows * cols)

    start_state = (start[0], start[1], 0)
    open_list = []
    counter = 0
    heapq.heappush(open_list,(heuristic(start, goal), counter, start_state))
    g_score = {start_state: 0}
    came_from = {start_state: None}
    closed_set = set()
    nodes_expanded = 0
    max_frontier_size = 1

    while open_list:
        _, _, current_state = heapq.heappop(open_list)
        if current_state in closed_set:
            continue
        row, col, time_step = current_state
        current_cell = (row, col)
        if current_cell == goal:
            path = _reconstruct_path(came_from, current_state)
            return {
                "path": path,
                "path_length": len(path) - 1,
                "arrival_time": time_step,
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "success": True,
            }

        if time_step >= max_time:
            continue

        closed_set.add(current_state)
        nodes_expanded += 1
        next_time = time_step + 1

        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            next_cell = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[next_cell] == 1:
                continue

            if next_cell in reservations.get(next_time, set()):
                continue

            opposite_edge = (next_cell, current_cell)
            if opposite_edge in edge_reservations.get(next_time, set()):
                continue

            next_state = (nr, nc, next_time)
            if next_state in closed_set:
                continue

            tentative_g = g_score[current_state] + 1
            if tentative_g < g_score.get(next_state, float("inf")):
                came_from[next_state] = current_state
                g_score[next_state] = tentative_g
                f_score = tentative_g + heuristic(next_cell, goal)
                counter += 1
                heapq.heappush(open_list, (f_score, counter, next_state))

        max_frontier_size = max(max_frontier_size, len(open_list))

    return {
        "path": None,
        "path_length": None,
        "arrival_time": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "success": False,
    }


def reserve_path(path, reservations, edge_reservations, max_time):
    if not path:
        return

    for t, cell in enumerate(path):
        reservations[t].add(tuple(cell))

    for t in range(1, len(path)):
        previous_cell = tuple(path[t - 1])
        current_cell = tuple(path[t])
        edge_reservations[t].add((previous_cell, current_cell))

    goal_cell = tuple(path[-1])
    for t in range(len(path), max_time + 1):
        reservations[t].add(goal_cell)


def prioritized_cooperative_astar(grid, starts, goals, heuristic=manhattan_distance,max_time=None, priority_order=None):
    starts = [tuple(int(x) for x in point) for point in starts]
    goals = [tuple(int(x) for x in point) for point in goals]

    if len(starts) != len(goals):
        raise ValueError("starts and goals must contain the same number of robots.")
    if len(set(starts)) != len(starts):
        raise ValueError("Each robot must have a unique start cell.")
    if len(set(goals)) != len(goals):
        raise ValueError("Each robot must have a unique goal cell.")

    rows, cols = grid.shape
    if max_time is None:
        max_time = max(4 * rows * cols, 1)

    if priority_order is None:
        priority_order = list(range(len(starts)))
    else:
        priority_order = list(priority_order)
        if sorted(priority_order) != list(range(len(starts))):
            raise ValueError("priority_order must be a permutation of robot indices.")

    reservations = defaultdict(set)
    edge_reservations = defaultdict(set)
    results = {}

    for robot_index in priority_order:
        result = cooperative_astar(
            grid=grid,
            start=starts[robot_index],
            goal=goals[robot_index],
            reservations=reservations,
            edge_reservations=edge_reservations,
            heuristic=heuristic,
            max_time=max_time,
        )

        results[f"Robot {robot_index + 1}"] = result

        if result["success"]:
            reserve_path(
                result["path"],
                reservations,
                edge_reservations,
                max_time,
            )

    return {
        f"Robot {i + 1}": results[f"Robot {i + 1}"]
        for i in range(len(starts))
    }
