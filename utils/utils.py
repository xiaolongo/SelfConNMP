import math
import os.path as osp

import torch
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_data(dataset, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets')
    dataset = TUDataset(path, dataset, cleaned=cleaned)
    dataset.data.edge_attr = None
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    return dataset


def separate_data(dataset):
    # labels = [g.y for g in dataset]
    idx_list = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12345)
    # for idx in skf.split(np.zeros(len(labels)), labels):
    for idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        idx_list.append(idx)
    for fold_idx in range(10):
        train_idx, test_idx = idx_list[fold_idx]
        with open('train_idx-%d.txt' % (fold_idx + 1), 'w') as f:
            for index in train_idx:
                f.write(str(index))
                f.write('\n')
        with open('test_idx-%d.txt' % (fold_idx + 1), 'w') as f:
            for index in test_idx:
                f.write(str(index))
                f.write('\n')


def init_data(data):
    temp_data = []
    for i in range(len(data)):
        #  master node
        master_node = torch.mean(data[i].x, dim=0).unsqueeze(0)
        temp_x = torch.cat((data[i].x, master_node), dim=0)

        # edge_index_master
        row, col = data[i].edge_index
        row_m = torch.full((max(row).item() + 2, ),
                           max(row) + 1,
                           dtype=torch.long)
        col_m = torch.arange(max(row) + 2, dtype=torch.long)

        row_new = torch.cat((row, row_m), dim=0).unsqueeze(0)
        col_new = torch.cat((col, col_m), dim=0).unsqueeze(0)

        temp_edge_index = torch.cat((row_new, col_new), dim=0)

        temp_data_i = Data(x=temp_x, edge_index=temp_edge_index, y=data[i].y)
        temp_data.append(temp_data_i)
    return temp_data


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


if __name__ == '__main__':
    dataset = load_data('REDDIT-MULTI-5K')
    separate_data(dataset)
