"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from .distances import euclidean_dist


def k_neigh_index(rank, i, k1):
    forward_k_index = rank[i, :k1+1]
    backward_k_index = rank[forward_k_index, :k1+1]
    fi = torch.where(backward_k_index == i)[0]
    return forward_k_index[fi]


def re_ranking(qf: Tensor, gf: Tensor, k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> Tensor:
    features = torch.cat([qf, gf])
    N = features.shape[0]
    dist = euclidean_dist(features, features)
    dist = (dist / torch.max(dist, dim=0)[0]).t()
    V = torch.zeros_like(dist)
    initial_rank = torch.argsort(dist)

    for i in range(N):
        k_reciprocal_index = k_neigh_index(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index

        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = k_neigh_index(initial_rank, candidate, int(round(k1//2)))

            if len(np.intersect1d(candidate_k_reciprocal_index.numpy(), k_reciprocal_index.numpy())) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.cat([k_reciprocal_expansion_index, candidate_k_reciprocal_index], dim=0)

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist[i, k_reciprocal_expansion_index], dim=-1)
    
    dist = dist[:qf.shape[0],]

    if k2 != 1:
        V_qe = torch.zeros_like(V)
        for i in range(N):
            V_qe[i, :] = V[initial_rank[i, :k2], :].mean(dim=0)
        V = V_qe
    del V_qe, initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(torch.where(V[:, i] != 0)[0])

    jaccard_dist = torch.zeros_like(dist)

    for i in range(qf.shape[0]):
        temp_min = torch.zeros((1, N))
        indNonZeros = torch.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZeros]

        for indImage, indNonZero in zip(indImages, indNonZeros):
            temp_min[0, indImage] += torch.minimum(V[i, indNonZero], V[indImage, indNonZero])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + dist * lambda_value
    del dist, V, jaccard_dist
    return final_dist[:qf.shape[0], qf.shape[0]:]


if __name__ == '__main__':
    from torch.nn import functional as F
    query_features = F.normalize(torch.randn(4, 2048))
    gallery_features = F.normalize(torch.randn(4, 2048))
    metric = re_ranking(query_features, gallery_features)
    print(metric.shape)