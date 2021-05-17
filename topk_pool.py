import argparse
import os.path as osp

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GraphConv, JumpingKnowledge, TopKPooling,
                                global_add_pool)

from utils.utils import load_data

# from torch.utils.tensorboard import SummaryWriter

setproctitle.setproctitle('fxl')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--idx', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.01)
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# writer = SummaryWriter('runs/la_PROTEINS')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = load_data(args.dataset)
index_train = []
index_test = []
with open(
        osp.join(osp.dirname(osp.realpath(__file__)), 'datasets',
                 '%s' % args.dataset, '10fold_idx',
                 'train_idx-%d.txt' % args.idx), 'r') as f_train:
    for line in f_train:
        index_train.append(int(line.split('\n')[0]))
with open(
        osp.join(osp.dirname(osp.realpath(__file__)), 'datasets',
                 '%s' % args.dataset, '10fold_idx',
                 'test_idx-%d.txt' % args.idx), 'r') as f_test:
    for line in f_test:
        index_test.append(int(line.split('\n')[0]))

train_dataset = [dataset[i] for i in index_train]
test_dataset = [dataset[j] for j in index_test]

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden = args.hidden
        num_layers = 5
        ratio = 0.8
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='add')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='add')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [TopKPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_add_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_add_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x,
                                                     edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    # writer.add_scalar('training loss', loss_all / len(train_dataset), epoch)
    # writer.flush()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.4f}, '
          'Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
    # print('{:.4f}'.format(train_loss))
