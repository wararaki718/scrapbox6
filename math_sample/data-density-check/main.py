import random

import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import train_test_split

from metrics import evaluate
from preprocessor import RatingPreprocessor


def experiment(user_ratings: sps.csr_matrix, density: float=1.0) -> dict:
    train_ratings, test_ratings = train_test_split(user_ratings, random_state=42)
    
    if density < 1.0:
        n_sample = int(len(train_ratings.data) * (1.0 - density))
        indices = random.sample(list(range(len(train_ratings.data))), n_sample)
        train_ratings.data[indices] = 0
        train_ratings.eliminate_zeros()

    # user recommendation
    model = AlternatingLeastSquares()
    model.fit(train_ratings)

    # evaluation
    scores = evaluate(model, train_ratings, test_ratings)
    return scores


def main() -> None:
    titles, ratings = get_movielens(variant="100k")
    print(titles.shape)
    print(ratings.shape)

    preprocessor = RatingPreprocessor()
    ratings = preprocessor.transform(ratings)
    print(ratings.shape)

    user_ratings = ratings.transpose().tocsr()
    print(user_ratings.shape)

    # experiment
    for density in [0.1, 0.3, 0.5, 0.7, 1.0]:
        score = experiment(user_ratings, density=density)
        print(score)
    print("DONE")


if __name__ == "__main__":
    main()
