from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import parquet


def main() -> None:
    a = np.array([1, 2, 3])
    print(a)
    print(type(a))

    filepath = Path("sample.parquet")
    parquet.write_table(pa.Table.from_arrays([a], names=["sample"]), filepath)
    print(filepath.exists())

    b = parquet.read_table(filepath)
    print(b)
    print(type(b))
    print(b.to_pandas().values)

    filepath.unlink()
    print(filepath.exists())

    print("DONE")


if __name__ == "__main__":
    main()
