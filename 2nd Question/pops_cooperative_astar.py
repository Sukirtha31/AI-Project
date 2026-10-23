"""Priority-Order Portfolio Search (POPS) for Cooperative A*.

Tests multiple priority orders and selects the best tested schedule.
This is not a guarantee of globally optimal MAPF makespan.
"""
from itertools import permutations
import random
from cooperative_astar import prioritized_cooperative_astar


def generate_priority_orders(n, max_orders=120, seed=42):
    if n <= 0:
        return []
    if n <= 4:
        return [list(p) for p in permutations(range(n))]
    rng = random.Random(seed)
    orders, seen = [], set()

    def add(order):
        key = tuple(order)
        if key not in seen and len(orders) < max_orders:
            seen.add(key)
            orders.append(list(order))

    add(list(range(n)))
    add(list(reversed(range(n))))
    for shift in range(1, n):
        add(list(range(shift, n)) + list(range(shift)))
    attempts = 0
    while len(orders) < max_orders and attempts < max_orders * 30:
        order = list(range(n))
        rng.shuffle(order)
        add(order)
        attempts += 1
    return orders


def prioritized_order_portfolio_search(grid, starts, goals, max_orders=120,
                                       seed=42, heuristic=None, max_time=None):
    n = len(starts)
    if n != len(goals):
        raise ValueError("starts and goals must have the same number of robots.")
    if n == 0:
        raise ValueError("At least one robot is required.")

    orders = generate_priority_orders(n, max_orders, seed)
    trials = []
    best_key, best_results, best_order, best_trial = None, None, None, None

    for order in orders:
        kwargs = {"priority_order": order}
        if heuristic is not None:
            kwargs["heuristic"] = heuristic
        if max_time is not None:
            kwargs["max_time"] = max_time
        results = prioritized_cooperative_astar(grid, starts, goals, **kwargs)
        all_success = all(results[f"Robot {i+1}"]["success"] for i in range(n))
        success_count = sum(results[f"Robot {i+1}"]["success"] for i in range(n))
        arrival = [results[f"Robot {i+1}"]["arrival_time"] for i in range(n)
                   if results[f"Robot {i+1}"]["success"]]
        lengths = [results[f"Robot {i+1}"]["path_length"] for i in range(n)
                   if results[f"Robot {i+1}"]["success"]]
        makespan = max(arrival) if all_success and arrival else float("inf")
        total_length = sum(lengths) if all_success else float("inf")
        trial = {
            "priority_order": [i+1 for i in order],
            "all_success": all_success,
            "success_count": success_count,
            "makespan": makespan,
            "total_path_length": total_length,
            "results": results,
        }
        trials.append(trial)
        key = (0 if all_success else 1,
               0 if all_success else -success_count,
               makespan, total_length)
        if best_key is None or key < best_key:
            best_key, best_results, best_order, best_trial = key, results, order, trial

    return {
        "results": best_results,
        "best_priority_order": [i+1 for i in best_order],
        "best_makespan": best_trial["makespan"],
        "best_total_path_length": best_trial["total_path_length"],
        "all_success": best_trial["all_success"],
        "orders_tested": len(orders),
        "trials": trials,
    }
