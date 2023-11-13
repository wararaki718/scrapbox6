import os
import shutil
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from environment import MazeEnvironment
from evaluator import MazeControllerEvaluator
from nn import FeedForwardNetwork
from parallel import EvaluatorParallel


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


def main() -> None:
    args = parse_args()
    save_path = Path("./out") / args.task

    initialize_experiment(args.name, save_path, args)

    envs = MazeEnvironment.read_environment(".", args.task)
    evaluator = MazeControllerEvaluator(envs, args.timesteps)

    parallel = EvaluatorParallel(
        num_workers=args.num_cores,
        evaluate_function=evaluator.evaluate_agent,
        decode_function=FeedForwardNetwork.create,
    )


    # create config

    print("DONE")


if __name__ == "__main__":
    main()
