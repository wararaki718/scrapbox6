import torch
import torch.nn as nn
import torch.nn.functional as F


# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
class ContrasiveLoss(nn.Module):
    def __init__(self, margin: float=1.0) -> None:
        super(ContrasiveLoss, self).__init__()
        self._margin = margin
    
    def forward(self, x_q: torch.Tensor, x_d: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distance = F.cosine_similarity(x_q, x_d)
        loss = torch.mean(
            (1 - y) * torch.pow(distance, 2) + y * torch.pow(torch.clamp(self._margin - distance, min=0.0), 2) / 2.0
        )
        return loss


if __name__ == "__main__":
    criterion = ContrasiveLoss()
    x_q = torch.randn(1, 128)
    x_d = torch.randn(10, 128)
    y = torch.randint(0, 1, (10,))
    
    loss = criterion(x_q, x_d, y)
    print(loss)
    