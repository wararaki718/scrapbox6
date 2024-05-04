from pathlib import Path

import pandas as pd


def main() -> None:
    msmarco_path = Path("/home/wararaki/workspace/splade/data/msmarco")
    score_path = msmarco_path / "hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"

    data = pd.read_pickle(score_path)
    print(type(data))
    print(len(data.keys()))
    print()
    print("check values:")
    keys = list(data.keys())
    print(keys[0])
    values = data[keys[0]]
    print(type(values))
    print(values)
    
    print("DONE")


if __name__ == "__main__":
    main()
