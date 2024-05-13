import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups


class NewsLoader:
    @classmethod
    def load(cls) -> pd.DataFrame:
        news_groups = fetch_20newsgroups(subset="train")
        df = pd.DataFrame({
            "category": news_groups.target,
            "text": news_groups.data,
        })
        return df
