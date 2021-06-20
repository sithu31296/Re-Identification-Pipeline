import torch
import re
from PIL import Image
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from copy import deepcopy
import numpy as np
import random


class Market1501(Dataset):
    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if self.split == 'train':
            data_path = self.root / 'bounding_box_train'
        elif self.split == 'query':
            data_path = self.root / 'query'
        else:
            data_path = self.root / 'bounding_box_test'

        self.data, self.pids, self.camids = self.get_data(data_path)
        self.print_dataset_info()
        self.num_classes = len(self.pids)
        self.pidsdict = {pid: i for i, pid in enumerate(self.pids)}

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.496, 0.456), (0.229, 0.256, 0.224))
        ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.split == 'train': pid = self.pidsdict[pid]

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.img_transform(img)
        return img, pid, camid
        

    def get_data(self, dir_path: Path):
        img_paths = dir_path.glob('*.jpg')
        pattern = re.compile(r'([-\d]+)_c(\d)')
        data = []
        pids = set()
        camids = set()

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(str(img_path)).groups())
            if pid == -1: continue      # junk images are just ignored
            assert 0 <= pid <= 1501     # pid==0 means background
            assert 1 <= camid <= 6
            camid -= 1                  # index starts from 0
            pids.add(pid)
            camids.add(camid)
            data.append((str(img_path), pid, camid))
        return data, pids, camids


    def print_dataset_info(self):        
        table = [[self.split, len(self.data), len(self.pids), len(self.camids)]]
        print(tabulate(table, headers=['Subset', 'Images', 'Person IDs', 'Cameras'], numalign='right'))
        print()


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dict = defaultdict(list)

        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dict[pid].append(index)

        self.pids = list(self.index_dict.keys())

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dict[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = deepcopy(self.index_dict[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, self.num_instances, replace=True)
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_pids = deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    market = Market1501('C:/Users/sithu/Documents/Datasets/Market-1501-v15.09.15', split='gallery', transform=None)
    dataloader = DataLoader(market, batch_size=8, num_workers=4, sampler=RandomIdentitySampler(market.data, 8, 2))
    # img, pid, camid = next(iter(dataloader))
    # print(img.shape)
    # print(pid, camid)
    for img, pid, camid in dataloader:
        print(pid)
        break