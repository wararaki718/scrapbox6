from copy import deepcopy
from pathlib import Path

from neat import DefaultGenome, Config

from config import MEConfig
from decoder import StructureDecoder
from drawer import RobotDrawer


def reset_genome(config: Config) -> DefaultGenome:
    genome: DefaultGenome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    return genome


def mutate_genome(genome: DefaultGenome, config: Config) -> DefaultGenome:
    new_genome = deepcopy(genome)
    new_genome.key = genome.key + 1
    new_genome.mutate(config.genome_config)
    return new_genome


def main() -> None:
    robot_size = (5, 5)

    decoder = StructureDecoder(robot_size)
    drawer = RobotDrawer(robot_size)

    config_path = Path("./config.cfg")
    config = MEConfig.load(config_path)

    genome = reset_genome(config)
    states = decoder.decode(genome, config.genome_config)
    drawer.draw(states)

    while True:
        print("put the operation (m: mutate, r: reset genome, else: finish): ", end="")
        operation = input()
        if operation == "m":
            genome = mutate_genome(genome, config)
        elif operation == "r":
            genome = reset_genome(config)
        else:
            break
        
        states = decoder.decode(genome, config.genome_config)
        drawer.draw(states)

    print("DONE")

if __name__ == "__main__":
    main()
