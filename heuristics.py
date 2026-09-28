import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import pandas as pd


def manhattan_distance(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def euclidean_distance(current, goal):
    return np.sqrt((current[0] - goal[0])**2 +(current[1] - goal[1])**2)

def chebyshev_distance(current, goal):
    return max(abs(current[0] - goal[0]),abs(current[1] - goal[1]))

heuristics = {
    "Manhattan": manhattan_distance,
    "Euclidean": euclidean_distance,
    "Chebyshev": chebyshev_distance
}