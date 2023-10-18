import numpy as np
import pandas as pd


class PropensityScorerMatcher:
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        caliper = df["Zscore"].std() * 0.2

        treatments = df[df.treat > 0][["user_id", "Zscore"]].reset_index(drop=True)
        controls = df[df.treat == 0][["user_id", "Zscore"]].reset_index(drop=True)
        control_scores = controls.Zscore.values

        results = []
        for row in treatments.itertuples():
            index = np.abs(control_scores - row.Zscore).argmin()
            
            if np.abs(row.Zscore - control_scores[index]) > caliper:
                continue

            result = {
                "treatment": row.user_id,
                "control": controls.user_id.iloc[index],
                "treatment_zscore": row.Zscore,
                "control_zscore": controls.Zscore.iloc[index],
            }
            results.append(result)
            control_scores[index] = -9999
        
        result_df = pd.DataFrame(results)
        result_df = result_df.assign(
            diff=np.abs(
                result_df.treatment_zscore - result_df.control_zscore
            )
        )

        return result_df
