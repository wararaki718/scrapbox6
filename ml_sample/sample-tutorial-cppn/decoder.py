from typing import Dict, Tuple

import neat
import numpy as np
from neat import DefaultGenome, Config



class StructureDecoder:
    def __init__(self, size: Tuple[int, int]) -> None:
        self._size = size

        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing="ij")
        x: np.ndarray = x.flatten()
        y: np.ndarray = y.flatten()

        center = (np.array(size) - 1) / 2
        d = np.sqrt(np.square(x  - center[0], np.square(y - center[1])))
        self._inputs = np.vstack([x, y, d]).T
        self._types = [
            "empty",
            "rigid",
            "soft",
            "horizontal",
            "vertical",
        ]

    
    def decode(self, genome: DefaultGenome, config: Config) -> Dict[Tuple[int, int], str]:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)

        print("( x,  y) :   empty   rigid   soft    hori    vert ")
        print("            ======  ======  ======  ======  ======")

        states = dict()
        for input_ in self._inputs:
            state = cppn.activate(input_)
            x = int(input_[0])
            y = int(input_[1])

            m = np.argmax(state)
            voxel_type = self._types[m]

            print(
                f"({x: =2}, {y: =2}) :  " +
                "  ".join(("*" if i==m else " ") + f"{v: =+.2f}" for i, v in enumerate(state)) +
                f"  ->  {voxel_type.rjust(10)}"
            )
            states[(int(input_[0]), int(input_[1]))] = voxel_type
        print()

        return states
