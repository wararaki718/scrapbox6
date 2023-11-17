import os
import shutil
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import neat
from environment import MazeEnvironment
from evaluator import MazeControllerEvaluator
from nn import FeedForwardNetwork
from parallel import EvaluatorParallel
from population import Population


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Maze NEAT experiment'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help='experiment name (default: "{task}")'
    )
    parser.add_argument(
        '-t', '--task',
        default='medium', type=str,
        help='maze name (default: medium, built on "envs/maze/maze_files/")'
    )

    parser.add_argument(
        '-p', '--pop-size',
        default=500, type=int,
        help='population size of NEAT (default: 500)'
    )
    parser.add_argument(
        '-g', '--generation',
        default=500, type=int,
        help='iterations of NEAT (default: 500)'
    )

    parser.add_argument(
        '--timesteps',
        default=400, type=int,
        help='limit of timestep for solving maze (default: 400)'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=4, type=int,
        help='number of parallel evaluation processes (default: 4)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true', default=False,
        help='not open window of progress figure (default: False)'
    )
    args = parser.parse_args()

    if args.name is None:
        args.name = args.task

    return args


def initialize_experiment(experiment_name: str, save_path: Path, args: Namespace) -> None:
    try:
        os.makedirs(save_path)
    except:
        print(f"THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS")
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            quit()
        print()

    argument_file = save_path / "arguments.json"
    with open(argument_file, "w") as f:
        json.dump(args.__dict__, f, indent=4)


def make_config(config_file: Path, extra_info=None, custom_config=None) -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
        extra_info=extra_info,
        custom_config=custom_config,
    )
    return config


def main() -> None:
    args = parse_args()
    save_path = Path("./out") / args.task

    # initialize_experiment(args.name, save_path, args)

    envs = MazeEnvironment.read_environment(".", args.task)
    evaluator = MazeControllerEvaluator(envs, args.timesteps)

    parallel = EvaluatorParallel(
        num_workers=args.num_cores,
        evaluate_function=evaluator.evaluate_agent,
        decode_function=FeedForwardNetwork.create,
    )

    # create config
    config_file = Path("./config")
    custom_config = [
        ("NEAT", "pop_size", args.pop_size),
    ]
    config = make_config(config_file, custom_config=custom_config)
    config_out_file = save_path / "maze_neat.cfg"
    config.save(config_out_file)

    population = Population(config)
    
    reporters = [
        neat.StdOutReporter(True),
    ]
    for reporter in reporters:
        population.add_reporter(reporter)
    
    population.run(fitness_function=parallel.evaluate, n=args.generation)

    print("DONE")


if __name__ == "__main__":
    main()
