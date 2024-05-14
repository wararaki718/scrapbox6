from implicit.als import AlternatingLeastSquares
from implicit.datasets.movielens import get_movielens
from implicit.evaluation import train_test_split, ndcg_at_k, mean_average_precision_at_k

from preprocessor import RatingPreprocessor


def main() -> None:
    titles, ratings = get_movielens(variant="100k")
    print(titles.shape)
    print(ratings.shape)

    preprocessor = RatingPreprocessor()
    ratings = preprocessor.transform(ratings)
    print(ratings.shape)

    user_ratings = ratings.transpose().tocsr()
    print(user_ratings.shape)

    train_ratings, test_ratings = train_test_split(user_ratings, random_state=42)

    # user recommendation
    model = AlternatingLeastSquares()
    model.fit(train_ratings)

    # evaluation
    k = 10
    ndcg_score = ndcg_at_k(model, train_ratings, test_ratings, k)
    map_score = mean_average_precision_at_k(model, train_ratings, test_ratings, k)
    print(ndcg_score)
    print(map_score)

    print("DONE")


if __name__ == "__main__":
    main()
