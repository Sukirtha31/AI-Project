from tracemalloc import start

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

def is_path_possible(grid, start, goal):
    rows, cols = grid.shape
    queue = deque([start])
    visited = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False

def generate_grid(rows=10, cols=10, obstacle_prob=0.25, no_of_robots=1):
    total_cells = rows * cols
    if 2 * no_of_robots > total_cells:
        raise ValueError("Not enough cells for unique start and goal positions.")
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        grid = np.random.choice([0, 1],size=(rows, cols),p=[1 - obstacle_prob, obstacle_prob])
        positions = random.sample(range(total_cells), 2 * no_of_robots)
        start = [
            (pos // cols, pos % cols)
            for pos in positions[:no_of_robots]
        ]
        goal = [
            (pos // cols, pos % cols)
            for pos in positions[no_of_robots:]
        ]
        for pos in start + goal:
            grid[pos] = 0

        all_paths_possible = all(is_path_possible(grid, start[i], goal[i]) for i in range(no_of_robots))
        if all_paths_possible:
            break

    else:
        raise RuntimeError(
            f"Could not generate a valid grid after {max_attempts} attempts. "
            "Try reducing obstacle_prob or the number of robots."
        )

    np.savetxt("results/grid.csv", grid, delimiter=",", fmt="%d")
    np.savetxt("results/start.csv", start, delimiter=",", fmt="%d")
    np.savetxt("results/goal.csv", goal, delimiter=",", fmt="%d")

    return grid, start, goal