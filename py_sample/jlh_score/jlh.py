from collections import Counter
from typing import List


class JLHScore:
    def compute(self, query_tokens: List[str], document_tokens: List[List[str]]) -> float:
        document_counters = Counter()
        for tokens in document_tokens:
            document_counters.update(tokens)

        hit_tokens = []
        for tokens in document_tokens:
            hits = set(tokens) & set(query_tokens)
            if hits:
                hit_tokens.append(tokens)

        hit_counters = Counter()
        for tokens in hit_tokens:
            hit_counters.update(tokens)
        
        p_foreground = sum([hit_counters.get(token, 0) for token in query_tokens]) / sum(hit_counters.values())
        p_background = sum([document_counters.get(token, 0) for token in query_tokens]) / sum(document_counters.values())

        return (p_foreground - p_background) * (p_foreground / p_background)
