import numpy as np
from fitness import AckleyFunctionFitness
from simulator import PSOSimulator


def main() -> None:
    n_particle = 100
    n_dim = 2
    X = np.random.random((n_particle, n_dim))
    V = np.random.random((n_particle, n_dim))

    fitness = AckleyFunctionFitness()
    simulator = PSOSimulator(fitness)
    print("simulate:")
    simulator.simulate(X, V)

    print("DONE")


if __name__ == "__main__":
    main()
