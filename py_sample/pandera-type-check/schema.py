from pandera import DataFrameModel


class RawUserSchema(DataFrameModel):
    user_id: int
    name: str
    age: int
    location: str


class UserSchema(DataFrameModel):
    user_id: int
    name: str
    age: int
    location: str
    area: str
