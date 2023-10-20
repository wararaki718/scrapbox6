import numpy as np
import pandas as pd


def ps_stratificasion(df: pd.DataFrame, step: float = 0.1) -> float:
    scores = np.arange(0, 1, step)

    results = np.array([])
    for score in scores:
        _df = df[
            (df.propensity_score >= score) &
            (df.propensity_score < score + step)
        ]
        t0 = np.array(_df[_df.special_course==0].score)
        t1 = np.array(_df[_df.special_course==1].score)

        if t0.size != 0 and t1.size != 0:
            results = np.append(results, t1.mean() - t0.mean())
    
    return results.mean()
