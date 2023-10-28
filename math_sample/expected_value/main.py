import numpy as np


def main() -> None:
    cards = np.array([1, 2, 3, 4])
    cards_2 = np.array([i**2 for i in cards])
    n_cards = np.array([1000, 500, 50, 10])
    
    Px = n_cards / n_cards.sum()
    Ex = (cards * Px).sum()

    Ex_2 = np.power(Ex, 2)
    E_x2 = (cards_2 * Px).sum()

    print(f"V(X)={E_x2 - Ex_2}")
    print("DONE")


if __name__ == "__main__":
    main()
