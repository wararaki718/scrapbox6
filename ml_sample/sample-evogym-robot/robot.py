from typing import Tuple

import numpy as np


def draw(probabilities: np.ndarray) -> int:
    probabilities_: np.ndarray = probabilities.copy()
    if type(probabilities_) != np.ndarray:
        probabilities_ = np.array(probabilities_)
    probabilities_ = probabilities_ / probabilities_.sum()

    threshold = np.random.uniform(0, 1)
    total = 0
    for i in range(probabilities_.size):
        total += probabilities_[i]
        if threshold <= total:
            return i


def recursive_search(x: int, y: int, connectivity: np.ndarray, robot: np.ndarray) -> None:
    if robot[x][y] == 0:
        return
    
    if connectivity[x][y] != 0:
        return
    
    connectivity[x][y] = 1
    for x_offset, y_offset in zip([0, 1, 0, -1], [1, 0, -1, 0]):
        if 0 <= (x + x_offset) < robot.shape[0] and 0 <= (y + y_offset) < robot.shape[1]:
            recursive_search(x + x_offset, y + y_offset, connectivity, robot)


def is_connected(robot: np.ndarray) -> bool:
    start: Tuple[int, int] = None
    for i in range(robot.shape[0]):
        if start:
            break

        for j in range(robot.shape[1]):
            if robot[i][j] != 0:
                start = (i, j)
                break
        
        if start == None:
            return False
    
    connectivity = np.zeros(robot.shape)
    recursive_search(start[0], start[1], connectivity, robot)
    
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] != 0 and connectivity[i][j] != 1:
                return False
    
    return True


def has_actuator(robot: np.ndarray) -> bool:
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i][j] == 3 or robot[i][j] == 4:
                return True
    return False


def get_full_connectivity(robot: np.ndarray) -> np.ndarray:
    output = []
    for i in range(robot.size):
        x = i % robot.shape[1]
        y = i // robot.shape[1]

        if robot[y][x] == 0:
            continue

        nx = x + 1
        ny = y

        if 0 <= nx < robot.shape[1] and 0 <= ny < robot.shape[0] and robot[ny][nx] != 0:
            output.append([x + robot.shape[1] * y, nx + robot.shape[1] * ny])
        
        nx = x
        ny = y + 1

        if 0 <= nx < robot.shape[1] and 0 <= ny < robot.shape[0] and robot[ny][nx] != 0:
            output.append([x + robot.shape[1] * y, nx + robot.shape[1] * ny])
    
    if len(output) == 0:
        return np.empty((0, 2)).T
    
    return np.array(output).T
