import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from schema import RawUserSchema, UserSchema


class UserLoader:
    @classmethod
    @pa.check_types(inplace=True)
    def load(cls) -> DataFrame[RawUserSchema]:
        df: DataFrame[RawUserSchema] = pd.DataFrame([
            [1, "hello", 10, "tokyo"],
            [2, "test", 20, "osaka"],
            [3, "sample", 30, "tokyo"],
        ], columns=["user_id", "name", "age", "location"])
        return df


class ErrorUserLoader:
    @classmethod
    @pa.check_types(inplace=True)
    def load(cls) -> DataFrame[UserSchema]:  # the return type is not matched!
        df: DataFrame[RawUserSchema] = pd.DataFrame([
            [1, "hello", 10, "tokyo"],
            [2, "test", 20, "osaka"],
            [3, "sample", 30, "tokyo"],
        ], columns=["user_id", "name", "age", "location"])
        return df
