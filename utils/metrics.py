from math import cos
from operator import pos
import torch
from torch import Tensor
import numpy as np
from torch.nn import functional as F
from .reranking import re_ranking
from .distances import consine_dist



def evaluate(dist, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    if dist.shape[1] < max_rank:
        max_rank = dist.shape[1]
        print("Note: number of gallery samples is quite small.")

    indices = torch.argsort(dist, dim=1)
    matches = g_pids[indices] == q_pids.unsqueeze(1)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.        # number of valid query

    for q_pid, q_camid, match, index in zip(q_pids, q_camids, matches, indices):
        # remove gallery samples that have the same pid and camid with query
        # only keep gallery samples that does not have the same pid and camid with query
        remove = (g_pids[index] == q_pid) & (g_camids[index] == q_camid)
        raw_cmc = match[~remove]
        if not torch.any(raw_cmc): continue        # this condition is true when query identity does not appear in gallery
        CMC = raw_cmc.cumsum(dim=0)

        pos_idx = torch.where(raw_cmc == 1)[0]
        max_pos_idx = torch.max(pos_idx)
        inp = CMC[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        CMC[CMC > 1] = 1
        all_cmc.append(CMC[:max_rank])

        num_valid_q += 1.

        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum(dim=0)
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = torch.tensor(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = torch.vstack(all_cmc).sum(0) / num_valid_q
    mAP = torch.tensor(all_AP).mean() * 100
    INP = torch.tensor(all_INP).mean() * 100

    ranks = [all_cmc[r-1]*100 for r in [1, 5, 10]]

    return ranks, mAP, INP
    

class R1_mAP:
    def __init__(self, max_rank=50) -> None:
        self.max_rank = max_rank

    def compute(self, 
        query_features: Tensor, query_pids: Tensor, query_camids: Tensor,
        gallery_features: Tensor, gallery_pids: Tensor, gallery_camids: Tensor
    ):
        dist = consine_dist(query_features, gallery_features)
        ranks, mAP, INP = evaluate(dist, query_pids, gallery_pids, query_camids, gallery_camids, self.max_rank)
        return ranks, mAP, INP


class R1_mAP_rerank:
    def __init__(self, max_rank=50) -> None:
        self.max_rank = max_rank

    def compute(self, 
        query_features: Tensor, query_pids: Tensor, query_camids: Tensor,
        gallery_features: Tensor, gallery_pids: Tensor, gallery_camids: Tensor
    ):
        query_features = F.normalize(query_features)
        gallery_features = F.normalize(gallery_features)
        
        dist = re_ranking(query_features, gallery_features, k1=20, k2=6, lambda_value=0.3)
        ranks, mAP, INP = evaluate(dist, query_pids, gallery_pids, query_camids, gallery_camids, self.max_rank)
        return ranks, mAP, INP


if __name__ == '__main__':
    metric = R1_mAP_rerank()
    torch.manual_seed(123)
    query_features = torch.randn(10, 2048)
    query_pids = torch.randint(high=751, size=(10,))
    query_camids = torch.randint(high=5, size=(10,))
    gallery_features = torch.randn(200, 2048)
    gallery_pids = torch.randint(high=751, size=(200,))
    gallery_camids = torch.randint(high=5, size=(200,))

    cmc, mAP, INP = metric.compute(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids)
    print(cmc[:5], mAP, INP)