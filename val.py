import os
import time
import torch
import argparse
import yaml

from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader

from models import choose_models
from datasets.market1501 import Market1501
from datasets.augmentations import get_transforms
from utils.metrics import R1_mAP, R1_mAP_rerank



@torch.no_grad()
def evaluate(model, query_loader, gallery_loader, metric, device):
    print('Evaluating...')
    model.eval()
    query_features, query_pids, query_camids = [], [], []
    gallery_features, gallery_pids, gallery_camids = [], [], []

    print('Predicting Query Features...')

    for iter, (img, pids, camids) in enumerate(query_loader):
        img = img.to(device)
        feats = model(img)
        query_features.append(feats)
        query_pids.append(pids)
        query_camids.append(camids)

    print('Predicting Gallery Features...')

    for iter, (img, pids, camids) in enumerate(gallery_loader):
        img = img.to(device)
        feats = model(img)
        gallery_features.append(feats)
        gallery_pids.append(pids)
        gallery_camids.append(camids)

    
    query_features = torch.cat(query_features, dim=0)
    query_pids = torch.cat(query_pids, dim=0)
    query_camids = torch.cat(query_camids, dim=0)
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_pids = torch.cat(gallery_pids, dim=0)
    gallery_camids = torch.cat(gallery_camids, dim=0)

    print('Calculating Metrics...')
    ranks, mAP, INP = metric.compute(query_features, query_pids, query_camids, gallery_features, gallery_pids, gallery_camids)

    return ranks, mAP, INP


def main(cfg):
    device = torch.device(cfg['DEVICE'])
    save_dir = Path(cfg['SAVE_DIR'])

    _, test_transform = get_transforms(cfg)

    query_dataset = Market1501(cfg['DATASET']['ROOT'], split='query', transform=test_transform)
    gallery_dataset = Market1501(cfg['DATASET']['ROOT'], split='gallery', transform=test_transform)

    query_loader = DataLoader(query_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    model = choose_models(gallery_dataset.num_classes, cfg)
    model.load_state_dict(torch.load(save_dir / f"{cfg['MODEL']['NAME']}.pth", map_location='cpu'))
    model = model.to(device)
    
    if cfg['EVAL']['RE_RANK']:
        print('Re-Ranking Enabled')
        metric = R1_mAP_rerank(cfg['EVAL']['MAX_RANK'])
    else:
        metric = R1_mAP(cfg['EVAL']['MAX_RANK'])
            
    ranks, mAP, INP = evaluate(model, query_loader, gallery_loader, metric, device)

    table = [
        ['Rank-1', ranks[0]],
        ['Rank-5', ranks[1]],
        ['Rank-10', ranks[2]],
        ['mAP', mAP],
        ['INP', INP]
    ]

    print(tabulate(table, numalign='right'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)