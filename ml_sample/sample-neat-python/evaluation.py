from typing import List, Tuple
import neat


class Evaluator:
    def __init__(self, X_in: list, X_out: list) -> None:
        self._X_in = X_in
        self._X_out = X_out

    def evaluate(self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config, generation: int) -> None:
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            for x_in, x_out in zip(self._X_in, self._X_out):
                output = network.activate(x_in)
                genome.fitness -= (output[0] - x_out) ** 2
