import neat

from evaluation import Evaluator

def main() -> None:
    xor_inputs = [
        (0., 0.),
        (0., 1.),
        (1., 0.),
        (1., 1.),
    ]
    xor_outputs = [
        (0.), (0.), (0.), (1.),
    ]

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-feedforward",
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(False))

    evaluator = Evaluator(xor_inputs, xor_outputs)
    
    winner = population.run(evaluator.evaluate)

    print(f"best genome: {winner}")
    print()

    print("output:")
    winner_network = neat.nn.FeedForwardNetwork.create(winner, config)
    for x_in, x_out in zip(xor_inputs, xor_outputs):
        output = winner_network.activate(x_in)
        print(f"input : {x_in}")
        print(f"output: {x_out}")
        print(f"expect: {output}")
        print()
    print("DONE")


if __name__ == "__main__":
    main()
