import torch
import torch.nn as nn


def main() -> None:
    y = torch.ones([10, 64], dtype=torch.float32)
    y_pred = torch.full([10, 64], 1.5)
    pos_weight = torch.ones([64])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(y_pred, y)
    print(loss)
    print("DONE")


if __name__ == "__main__":
    main()
