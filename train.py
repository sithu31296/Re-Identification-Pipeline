import os
import time
import torch
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models import choose_models
from utils import choose_loss
from datasets.market1501 import Market1501
from datasets.augmentations import get_transforms
from utils.metrics import R1_mAP
from .val import evaluate


def main(cfg):
    start = time.time()
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()
    epochs = cfg['TRAIN']['EPOCHS']
    loss_fn_name = cfg['TRAIN']['LOSS']['NAME']
    device = torch.device(cfg['DEVICE'])

    train_transform, test_transform = get_transforms(cfg)

    train_dataset = Market1501(cfg['DATASET']['ROOT'], split='train', transform=train_transform)
    query_dataset = Market1501(cfg['DATASET']['ROOT'], split='query', transform=test_transform)
    gallery_dataset = Market1501(cfg['DATASET']['ROOT'], split='gallery', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'] , shuffle=True, num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    num_classes = train_dataset.num_classes

    model = choose_models(num_classes, cfg)
    model = model.to(device)
    
    loss_fn = choose_loss(loss_fn_name, num_classes, cfg)
    optimizer = SGD(model.parameters(), cfg['TRAIN']['LR'])
    scheduler = StepLR(optimizer, cfg['TRAIN']['STEP_LR']['STEP_SIZE'], cfg['TRAIN']['STEP_LR']['GAMMA'])
    metric = R1_mAP(cfg['EVAL']['MAX_RANK'])

    best_rank, best_mAP = 0.0, 0.0
    best_ranks, best_INP = [], 0.0
    iters_per_epoch = int(len(train_dataset)) / cfg['TRAIN']['BATCH_SIZE']
    writer = SummaryWriter(save_dir / 'logs')

    for epoch in range(1, epochs+1):
        model.train()

        train_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")

        for iter, (img, lbl, _) in enumerate(train_loader):
            img = img.to(device)
            lbl = lbl.to(device)

            preds, feats = model(img)

            if loss_fn_name == 'ce':
                loss = loss_fn(preds, lbl)
            elif loss_fn_name == 'ce+triplet':
                loss = loss_fn(preds, feats, lbl)
            else:
                loss = loss_fn(feats, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss:.8f}")

        train_loss /= len(train_dataset)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)

        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch % cfg['TRAIN']['EVAL_INTERVAL'] == 0) and (epoch >= cfg['TRAIN']['EVAL_INTERVAL']):
            ranks, mAP, INP = evaluate(model, query_loader, gallery_loader, metric, device)

            print(f"Rank-1: {ranks[0]:>0.1f} Rank-5: {ranks[1]:>0.1f} Rank-10: {ranks[2]:>0.1f} mAP: {mAP:>0.1f} INP: {INP:>0.1f}")

            writer.add_scalar('val/rank1', ranks[0], epoch)
            writer.add_scalar('val/rank5', ranks[1], epoch)
            writer.add_scalar('val/rank10', ranks[2], epoch)

            if ranks[0] > best_rank and mAP > best_mAP:
                best_rank = ranks[0]
                best_mAP = mAP
                best_ranks = ranks
                best_INP = INP
                torch.save(model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}.pth")

                print(f"Best Rank-1: {ranks[0]:>0.1f} mAP: {mAP:>0.1f}")
                
    writer.close()
    pbar.close()

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Rank-1', best_rank],
        ['Rank-5', best_ranks[1]],
        ['Rank-10', best_ranks[2]],
        ['mAP', best_mAP],
        ['INP', best_INP],
        ['Total Training Time', total_time]
    ]

    print(tabulate(table, numalign='right'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)