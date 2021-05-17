import os.path as osp
import torch
import argparse
import setproctitle
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
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
# writer = SummaryWriter('runs/reddit_m/sc_reddit_m')
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
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = args.hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim),
                         ReLU(), torch.nn.BatchNorm1d(dim))
        self.conv1 = GINConv(nn1)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv2 = GINConv(nn2)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv3 = GINConv(nn3)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv4 = GINConv(nn4)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv5 = GINConv(nn5)

        self.fc1 = Sequential(Linear(num_features, dim), ReLU(),
                              torch.nn.BatchNorm1d(dim))
        self.fc2 = Sequential(Linear(dim, dim), ReLU(),
                              torch.nn.BatchNorm1d(dim))

        self.lin = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):

        x0_g = global_add_pool(x, batch)
        x0_g = self.fc1(x0_g)

        x1 = self.conv1(x, edge_index)
        x1_g = global_add_pool(x1, batch)
        x1_g = self.fc2(x0_g + x1_g)

        x2 = self.conv2(x1, edge_index)
        x2_g = global_add_pool(x2, batch)
        x2_g = self.fc2(x1_g + x2_g)

        x3 = self.conv3(x2, edge_index)
        x3_g = global_add_pool(x3, batch)
        x3_g = self.fc2(x2_g + x3_g)

        x4 = self.conv4(x3, edge_index)
        x4_g = global_add_pool(x4, batch)
        x4_g = self.fc2(x3_g + x4_g)

        x5 = self.conv5(x4, edge_index)
        x5_g = global_add_pool(x5, batch)
        x5_g = self.fc2(x4_g + x5_g)

        x = x0_g + x1_g + x2_g + x3_g + x4_g + x5_g
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

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
        output = model(data.x, data.edge_index, data.batch)
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
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
