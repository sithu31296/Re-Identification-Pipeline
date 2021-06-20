import torch
from torch import Tensor
from torch.nn import functional as F


def consine_dist(x: Tensor, y: Tensor) -> Tensor:
    x = F.normalize(x)
    y = F.normalize(y)
    return 2 - 2 * (x @ y.t())


def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
    xx, yy = torch.meshgrid((x**2).sum(1), (y**2).sum(1))
    return xx + yy - 2 * (x @ y.t())