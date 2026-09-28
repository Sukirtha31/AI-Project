import numpy as np
import heapq
import matplotlib.pyplot as plt

def astar(grid, start, goal, heuristic):
    start = tuple(int(x) for x in start)
    goal = tuple(int(x) for x in goal)
    rows, cols = grid.shape
    open_list = []
    counter = 0
    g_score = {start: 0}
    came_from = {}
    closed_set = set()

    h = heuristic(start, goal)
    heapq.heappush(open_list, (h, counter, start))
    nodes_expanded = 0
    max_frontier_size = 1

    directions = [(-1, 0),(1, 0),(0, -1),(0, 1)]

    while open_list:
        f_score, _, current = heapq.heappop(open_list)
        if current in closed_set:
            continue
        
        if current == goal:
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            return {
                "path": path,
                "path_length": len(path) - 1,
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier_size,
                "success": True
            }
        closed_set.add(current)
        nodes_expanded += 1
        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + 1           
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_list,(f, counter, neighbor))

        max_frontier_size = max(max_frontier_size,len(open_list))


    return {
        "path": None,
        "path_length": None,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "success": False
    }