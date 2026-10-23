"""Priority-Based Search (PBS) for grid MAPF.

PBS searches over partial priority constraints. On a path conflict between
robots i and j, it creates two branches: i has priority over j, or j has
priority over i. Paths are replanned under the constraints of higher-priority
robots. This implementation orders high-level nodes by (makespan, sum of path
lengths), so it searches for low-makespan schedules first, but it is NOT a
formal guarantee of globally optimal makespan. The finite time horizon and
priority-based restriction can also cause failures on solvable instances.
"""
from __future__ import annotations
import heapq
import itertools
import time
from collections import defaultdict, deque
import numpy as np
from cooperative_astar import cooperative_astar, manhattan_distance


def _topological_order(n, edges):
    """Return a topological ordering, or None if priority constraints cycle."""
    outgoing = [[] for _ in range(n)]
    indegree = [0] * n
    for high, low in edges:
        outgoing[high].append(low)
        indegree[low] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in outgoing[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return order if len(order) == n else None


def _ancestors(n, edges):
    parents = [set() for _ in range(n)]
    for high, low in edges:
        parents[low].add(high)
    changed = True
    while changed:
        changed = False
        for v in range(n):
            new = set(parents[v])
            for p in list(parents[v]):
                new |= parents[p]
            if new != parents[v]:
                parents[v] = new
                changed = True
    return parents


def _plan_for_constraints(grid, starts, goals, edges, max_time, heuristic):
    """Replan all agents in a topological order, reserving only ancestors."""
    n = len(starts)
    order = _topological_order(n, edges)
    if order is None:
        return None
    parents = _ancestors(n, edges)
    planned = {}
    for i in order:
        reservations = defaultdict(set)
        edge_reservations = defaultdict(set)
        # Only higher-priority ancestors constrain this agent.
        for ancestor in parents[i]:
            r = planned.get(ancestor)
            if not r or not r['success']:
                return None
            path = r['path']
            for t, cell in enumerate(path):
                reservations[t].add(tuple(cell))
            for t in range(1, len(path)):
                edge_reservations[t].add((tuple(path[t-1]), tuple(path[t])))
            for t in range(len(path), max_time + 1):
                reservations[t].add(tuple(path[-1]))
        r = cooperative_astar(grid, starts[i], goals[i], reservations,
                              edge_reservations, heuristic=heuristic,
                              max_time=max_time)
        if not r['success']:
            return None
        planned[i] = r
    return {f'Robot {i+1}': planned[i] for i in range(n)}


def _position(path, t):
    return tuple(path[min(t, len(path)-1)])


def _first_conflict(results, n):
    paths = [results[f'Robot {i+1}']['path'] for i in range(n)]
    horizon = max(len(p) for p in paths) if paths else 0
    for t in range(horizon):
        for i in range(n):
            pi = _position(paths[i], t)
            for j in range(i+1, n):
                pj = _position(paths[j], t)
                if pi == pj:
                    return (i, j, t, 'vertex')
                if t > 0:
                    prev_i, prev_j = _position(paths[i], t-1), _position(paths[j], t-1)
                    if prev_i == pj and prev_j == pi:
                        return (i, j, t, 'edge_swap')
    return None


def _metrics(results, n):
    rs = [results[f'Robot {i+1}'] for i in range(n)]
    arrivals = [r['arrival_time'] for r in rs]
    lengths = [r['path_length'] for r in rs]
    return max(arrivals, default=0), sum(lengths), sum(r['nodes_expanded'] for r in rs), max((r['max_frontier_size'] for r in rs), default=0)


def priority_based_search(grid, starts, goals, heuristic=manhattan_distance,
                          max_time=None, max_high_level_nodes=5000):
    """Run PBS; returns results and high-level search diagnostics."""
    starts = [tuple(map(int, p)) for p in starts]
    goals = [tuple(map(int, p)) for p in goals]
    if len(starts) != len(goals) or not starts:
        raise ValueError('starts and goals must have the same non-zero length')
    if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
        raise ValueError('Starts and goals must each be unique')
    rows, cols = grid.shape
    if max_time is None:
        max_time = max(4 * rows * cols, 1)

    started = time.perf_counter()
    counter = itertools.count()
    empty_edges = frozenset()
    initial = _plan_for_constraints(grid, starts, goals, empty_edges, max_time, heuristic)
    if initial is None:
        return {'results': None, 'success': False, 'makespan': None,
                'total_path_length': None, 'high_level_nodes': 0,
                'runtime_seconds': time.perf_counter()-started,
                'reason': 'At least one individual path was not found.'}

    m, cost, _, _ = _metrics(initial, len(starts))
    heap = [(m, cost, next(counter), empty_edges, initial)]
    seen = {empty_edges}
    expanded = 0

    while heap and expanded < max_high_level_nodes:
        _, _, _, edges, paths = heapq.heappop(heap)
        expanded += 1
        conflict = _first_conflict(paths, len(starts))
        if conflict is None:
            makespan, total, nodes, frontier = _metrics(paths, len(starts))
            return {'results': paths, 'success': True, 'makespan': makespan,
                    'total_path_length': total, 'nodes_expanded': nodes,
                    'max_frontier_size': frontier, 'high_level_nodes': expanded,
                    'runtime_seconds': time.perf_counter()-started,
                    'priority_constraints': [(a+1, b+1) for a,b in sorted(edges)],
                    'reason': 'Collision-free priority-constrained schedule found.'}
        i, j, _, _kind = conflict
        for high, low in ((i, j), (j, i)):
            new_edges = frozenset(set(edges) | {(high, low)})
            if new_edges in seen or _topological_order(len(starts), new_edges) is None:
                continue
            seen.add(new_edges)
            candidate = _plan_for_constraints(grid, starts, goals, new_edges, max_time, heuristic)
            if candidate is None:
                continue
            cm, cc, _, _ = _metrics(candidate, len(starts))
            heapq.heappush(heap, (cm, cc, next(counter), new_edges, candidate))

    return {'results': None, 'success': False, 'makespan': None,
            'total_path_length': None, 'high_level_nodes': expanded,
            'runtime_seconds': time.perf_counter()-started,
            'reason': f'No solution found within {expanded} high-level expansions and horizon {max_time}.'}
