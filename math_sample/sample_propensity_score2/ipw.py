import numpy as np
import pandas as pd


def ipw(df: pd.DataFrame) -> float:
    N = len(df)
    D = df.special_course
    Y = df.score
    e_X = df.propensity_score

    E_y1 = (1 / N) * np.sum((D / e_X) * Y)
    E_y0 = (1 / N) * np.sum(((1 - D) / (1 - e_X) * Y))

    return E_y1 - E_y0
