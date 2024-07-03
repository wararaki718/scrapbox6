import torch
import torch.nn as nn
import torch.nn.functional as F


# contrastive loss
# https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html

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


class TripletContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super(TripletContrastiveLoss, self).__init__()
        self._triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
        )
    
    def forward(self, x_q: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._triplet_loss(x_q, x_pos, x_neg)
        return loss


class TripletContrastiveWithDotLoss(nn.Module):
    def __init__(self) -> None:
        super(TripletContrastiveWithDotLoss, self).__init__()
        self._triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: torch.matmul(x, y.T)
        )
    
    def forward(self, x_q: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._triplet_loss(x_q, x_pos, x_neg)
        return loss


if __name__ == "__main__":
    criterion = ContrasiveLoss()
    x_q = torch.randn(1, 128)
    x_d = torch.randn(10, 128)
    y = torch.randint(0, 1, (10,))
    
    loss = criterion(x_q, x_d, y)
    print(loss)

    x_q = torch.randn(1, 128)
    x_pos = torch.randn(10, 128)
    x_neg = torch.randn(10, 128)

    triplet_criterion = TripletContrastiveLoss()
    loss = triplet_criterion(x_q, x_pos, x_neg)
    print(loss)

    triplet_dot_criterion = TripletContrastiveWithDotLoss()
    loss = triplet_criterion(x_q, x_pos, x_neg)
    print(loss)
