from typing import Dict, Tuple

import matplotlib.pyplot as plt


class RobotDrawer:
    def __init__(self, size: Tuple[int, int]) -> None:
        self._size = size
        fig, self._ax = plt.subplots(figsize=size)
        self._colors = {
            "empty": [1, 1, 1],
            "rigid": [0.15, 0.15, 0.15],
            "soft": [0.75, 0.75, 0.75],
            "horizontal": [0.93, 0.58, 0.31],
            "vertical": [0.49, 0.68, 0.83],
        }

    def draw(self, state: dict) -> Dict[Tuple[int, int], str]:
        self._ax.cla()
        for (x, y), voxel in state.items():
            color = self._colors[voxel]
            self._ax.fill_between([x, x+1], [y]*2, [y+1]*2, color=color)
        
        self._ax.grid()
        self._ax.set_xlim([0, self._size[0]])
        self._ax.set_ylim([0, self._size[1]])
        self._ax.invert_yaxis()

        plt.pause(0.01)
