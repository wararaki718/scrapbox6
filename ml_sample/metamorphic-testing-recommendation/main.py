import numpy as np
import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares
from implicit.datasets.movielens import get_movielens


def train(titles: np.ndarray, ratings: sps.csr_matrix) -> None:
    model = AlternatingLeastSquares(random_state=42)
    user_ratings = ratings.T.tocsr()

    model.fit(user_ratings)
    print("model trained")

    user_count = np.ediff1d(ratings.indptr)
    to_generate = sorted(np.arange(len(titles)), key=lambda x: -user_count[x])
    batch_size = 10
    for i in range(0, len(to_generate), batch_size):
        batch = to_generate[i:i+batch_size]
        movie_ids, scores = model.similar_items(batch, 11)
        for j, movie_id in enumerate(batch):
            if ratings.indptr[movie_id] != ratings.indptr[movie_id+1]:
                title = titles[movie_id]
                for other, score in zip(movie_ids[j][1:], scores[j][1:]):
                    print(f"{title}: movie_id={other}, {titles[other]} = {score}")
                break
        break
    print()


def main() -> None:
    # download
    titles, ratings = get_movielens("100k")
    print(titles.shape)
    print(ratings.shape)

    # preprocessing
    min_rating = 4.0
    ratings.data[ratings.data < min_rating] = 0.0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))
    print(ratings.shape)

    # recommendation
    print("[original]")
    train(titles, ratings)

    # metamorphic-testing
    movie_id = 181
    print(f"remove movie_id={movie_id}: {titles[movie_id]}")
    print()

    fix_titles: np.ndarray = np.delete(titles, movie_id)
    print(fix_titles.shape)

    fix_ratings = sps.csr_matrix(np.delete(ratings.toarray(), movie_id, 0))
    print(fix_ratings.shape)

    print("[metamorphic-testing]")
    train(fix_titles, fix_ratings)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
