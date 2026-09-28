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

def generate_grid(rows=10, cols=10, obstacle_prob=0.25):
    grid = np.random.choice([0, 1], size=(rows, cols),p=[1 - obstacle_prob, obstacle_prob])

    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    goal = (random.randint(0, rows - 1),random.randint(0, cols - 1))
    while goal == start:
        goal = (random.randint(0, rows - 1),random.randint(0, cols - 1))
    grid[start] = 0
    grid[goal] = 0

    while not is_path_possible(grid, start, goal):
        grid = np.random.choice([0, 1],size=(rows, cols),p=[1 - obstacle_prob, obstacle_prob])
        grid[start] = 0
        grid[goal] = 0
    
    np.savetxt("results/grid.csv", grid, delimiter=",", fmt="%d")
    np.savetxt("results/start.csv", start, delimiter=",", fmt="%d")
    np.savetxt("results/goal.csv", goal, delimiter=",", fmt="%d")

    return grid, start, goal