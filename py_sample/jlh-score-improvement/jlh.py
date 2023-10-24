from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class JLHScore:
    def compute(self, queries: List[str], documents: List[str], verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        vectorizer = CountVectorizer(analyzer=lambda x: x.split())

        # documents
        X_documents = vectorizer.fit_transform(documents)
        X_documents_sum = X_documents.sum(axis=0)
        total_occurence_rate = X_documents_sum / X_documents.shape[0]

        # queries
        X_queries = vectorizer.transform(queries)
        X_queries_sum = X_queries.sum(axis=0)
        group_occurence_rate = X_queries_sum / X_queries.shape[0]

        # compute score
        sub = (group_occurence_rate - total_occurence_rate)
        div = (group_occurence_rate / total_occurence_rate)
        scores = np.squeeze(np.asarray(np.multiply(sub, div)))

        if verbose:
            for score, token in zip(scores, vectorizer.get_feature_names_out()):
                print(f"{token}: {score}")
            print()

        return scores, vectorizer.get_feature_names_out()
