import numpy as np
import scipy.sparse as sps
from implicit.nearest_neighbours import bm25_weight


class RatingPreprocessor:
    def transform(self, ratings: sps.csr_matrix, threshold: float=4.0) -> sps.csr_matrix:
        ratings.data[ratings.data < threshold] = 0
        ratings.eliminate_zeros()
        ratings.data = np.ones(len(ratings.data))

        ratings = bm25_weight(ratings, B=0.9).tocsr()

        return ratings
