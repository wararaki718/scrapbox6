import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ndcg_at_k, mean_average_precision_at_k


def evaluate(
    model: AlternatingLeastSquares,
    train_ratings: sps.csr_matrix,
    test_ratings: sps.csr_matrix,
    k: int=10
) -> dict:
    ndcg_score = ndcg_at_k(model, train_ratings, test_ratings, k)
    map_score = mean_average_precision_at_k(model, train_ratings, test_ratings, k)
    return {
        "ndcg": ndcg_score,
        "map": map_score,
    }
