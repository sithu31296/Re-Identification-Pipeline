import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple
from .distances import consine_dist, euclidean_dist


class CircleLoss(nn.Module):
    #from https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py 
    def __init__(self, margin: float, gamma: float) -> None:
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.delta_p = 1 - self.margin
        self.delta_n = self.margin
        self.softplus = nn.Softplus()

    def forward(self, features: Tensor, targets: Tensor) -> Tensor:
        features = F.normalize(features)
        # normed_features = F.normalize(features)
        # features = features.div(normed_features.expand_as(features))
        dist = (features @ features.t()).flatten()
        label_matrix = targets.unsqueeze(1) == targets.unsqueeze(0)
        positive_matrix = label_matrix.triu(diagonal=1).flatten()
        negative_matrix = ~label_matrix.triu(diagonal=1).flatten()

        features = dist[positive_matrix]
        targets = dist[negative_matrix]

        alpha_p = torch.clamp_min(- features + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(targets + self.margin, min=0.)

        logit_p = - alpha_p * (features - self.delta_p) * self.gamma
        logit_n = alpha_n * (targets - self.delta_n) * self.gamma

        loss = self.softplus(torch.logsumexp(logit_n, dim=-1) + torch.logsumexp(logit_p, dim=-1))
        return loss


class TripletLoss(nn.Module):
    """
    Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'
    """
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, features: Tensor, targets: Tensor) -> Tensor:
        dist = euclidean_dist(features, features)

        is_pos = targets.unsqueeze(0) == targets.unsqueeze(1)
        is_neg = ~is_pos

        dist_ap, dist_an = self.hard_example_mining(dist, is_pos, is_neg)
        y = torch.ones_like(dist_an)

        if self.margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            if loss == float('inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, 0.3)
        return loss

    def hard_example_mining(self, dist, is_pos, is_neg):
        """For each anchor, find the hardest positive and negative sample.
        Args:
            dist_mat: pair wise distance between samples, shape [N, M]
            is_pos: positive index with shape [N, M]
            is_neg: negative index with shape [N, M]
        Returns:
            dist_ap: Tensor, distance(anchor, positive); shape [N]
            dist_an: Tensor, distance(anchor, negative); shape [N]
        NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
        """
        assert len(dist.size()) == 2

        dist_ap = torch.max(dist * is_pos, dim=1)[0]
        dist_an = torch.min(dist * is_neg + is_pos * 1e9, dim=1)[0]
        return dist_ap, dist_an

    def weighted_example_mining(self, dist, is_pos, is_neg):
        """For each anchor, find the weighted positive and negative sample.
        """
        assert len(dist.size()) == 2

        dist_ap = dist*is_pos
        dist_an = dist*is_neg

        weights_ap = self.softmax_weights(dist_ap, is_pos)
        weights_an = self.softmax_weights(-dist_an, is_neg)

        dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
        dist_an = torch.sum(dist_an * weights_an, dim=1)

        return dist_ap, dist_an


    def softmax_weights(self, dist, mask):
        max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
        diff = dist - max_v
        Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6   # avoid division by zero
        W = torch.exp(diff) * mask / Z
        return W


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss with Label Smoothing Reularizer

    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_once(self, preds, targets):
        preds = self.logsoftmax(preds)
        targets = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets*preds).mean(0).sum()
        return loss

    def forward(self, preds, targets):
        if isinstance(preds, list):     # for PCB model only
            loss = self.forward_once(preds[0], targets)
            for i in range(5):
                loss += self.forward_once(preds[i+1], targets)
            return loss

        return self.forward_once(preds, targets)
        

    
class CombineLoss(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, margin=0.3):
        super().__init__()
        self.entropy = CrossEntropyLoss(num_classes, epsilon)
        self.triplet = TripletLoss(margin)

    def forward(self, preds: Tensor, features: Tensor, targets: Tensor) -> Tensor:
        return self.entropy(preds, targets) + self.triplet(features, targets)



if __name__ == "__main__":
    torch.manual_seed(123)
    feat = torch.rand(256, 64, requires_grad=True)
    lbl = torch.randint(high=10, size=(256,))
    criterion = TripletLoss(0.3)
    loss = criterion(feat, lbl)
    print(loss)