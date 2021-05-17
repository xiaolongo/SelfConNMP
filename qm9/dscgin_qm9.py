import argparse
import os.path as osp

import setproctitle
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GINConv, global_add_pool

setproctitle.setproctitle('fxl')
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=16)
parser.add_argument('--dim', type=int, default=64)
args = parser.parse_args()


class MyTransform:
    def __call__(self, data):
        data.y = data.y[:, args.target]
        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', 'QM9')
transform = T.Compose([MyTransform(), T.Distance()])
dataset = QM9(path, transform=transform).shuffle()

# Normalize targets to mean=0 and std=1
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, args.target].item(), std[:, args.target].item()

# dataset split
tenpercent = int(len(dataset) * 0.1)
test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]

test_loader = DataLoader(test_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = args.dim

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim),
                         ReLU(), torch.nn.BatchNorm1d(dim))
        self.conv1 = GINConv(nn1)
        self.fc1 = Linear(dim, dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv2 = GINConv(nn2)
        self.fc2 = Linear(dim, dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv3 = GINConv(nn3)
        self.fc3 = Linear(dim, dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv4 = GINConv(nn4)
        self.fc4 = Linear(dim, dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim), ReLU(),
                         torch.nn.BatchNorm1d(dim))
        self.conv5 = GINConv(nn5)
        self.fc5 = Linear(dim, dim)

        self.lin2 = Linear(dim, 1)

    def forward(self, data):
        x1 = self.conv1(data.x, data.edge_index)
        x1_g = global_add_pool(x1, data.batch)
        x1_g = F.relu(self.fc1(x1_g))

        x2 = self.conv2(x1, data.edge_index)
        x2_g = global_add_pool(x2, data.batch)
        x2_g = F.relu(self.fc2(x1_g + x2_g))

        x3 = self.conv3(x2, data.edge_index)
        x3_g = global_add_pool(x3, data.batch)
        x3_g = F.relu(self.fc3(x1_g + x2_g + x3_g))

        x4 = self.conv4(x3, data.edge_index)
        x4_g = global_add_pool(x4, data.batch)
        x4_g = F.relu(self.fc4(x1_g + x2_g + x3_g + x4_g))

        x5 = self.conv5(x4, data.edge_index)
        x5_g = global_add_pool(x5, data.batch)
        x5_g = F.relu(self.fc5(x1_g + x2_g + x3_g + x4_g + x5_g))

        x = self.lin2(x5_g)
        return x.view(-1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.7,
                                                       patience=5,
                                                       min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    # # get param
    # total_num = sum(p.numel() for p in model.parameters())
    # print('param:{}'.format(total_num))

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:4f}, Loss: {:.4f}, Validation MAE: {:.5f}, '
          'Test MAE: {:.5f}'.format(epoch, lr, loss, val_error, test_error))
