from typing import Optional, Tuple

import numpy as np

from robot import draw, is_connected, has_actuator, get_full_connectivity


class RobotFactory:
    @classmethod
    def create(cls, height: int, weight: int, probabilities: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        done = False
        if probabilities is None:
            n_probabilities = 5 # (empty, rigid, soft, h_act, v_act)
            probabilities = np.ones((n_probabilities)) / n_probabilities
            probabilities[0] = 0.6

        while (not done):
            robot = np.zeros((height, weight))
            for i in range(height):
                for j in range(weight):
                    robot[i][j] = draw(probabilities)

            if is_connected(robot) and has_actuator(robot):
                done = True
        
        return robot, get_full_connectivity(robot)
