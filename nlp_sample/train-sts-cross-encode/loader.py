import csv
import gzip
from pathlib import Path

from sentence_transformers import InputExample


class Loader:
    def __init__(self) -> None:
        pass

    @classmethod
    def load(cls, filepath: Path) -> tuple:
        train_samples = []
        vaild_samples = []
        test_samples = []

        with gzip.open(filepath, "rt", encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row["score"]) / 5.0

                input_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)
                if row["split"] == "dev":
                    vaild_samples.append(input_example)
                elif row["split"] == "test":
                    test_samples.append(input_example)
                else:
                    train_samples.append(input_example)
                    train_samples.append(InputExample(texts=[row["sentence2"], row["sentence1"]], label=score))

        return train_samples, vaild_samples, test_samples
