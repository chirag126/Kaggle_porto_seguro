from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

torch.manual_seed(212)
kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
train = pd.read_csv("/home/chirag212/Kaggle_porto_seguro/train.csv")
test = pd.read_csv("/home/chirag212/Kaggle_porto_seguro/test.csv")


X_train = train[train.columns[2:]]
X_test = test[test.columns[1:]]

y_train = train[train.columns[1]]

X_train = torch.from_numpy(np.array(X_train))
y_train = torch.from_numpy(np.array(y_train))

X_test = torch.from_numpy(np.array(X_test))
batch_size = 1

kwargs = {'num_workers': 1, 'pin_memory': True}
train = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size,
                                           shuffle=True, **kwargs)

validation = torch.utils.data.TensorDataset(X_test, torch.ones(X_test.size()[0]))
test_loader = torch.utils.data.DataLoader(dataset=validation, batch_size=batch_size,
                                         shuffle=False, **kwargs)


class cross_MLPNet(nn.Module):
    def __init__(self):
        super(cross_MLPNet, self).__init__()
        self.fc1 = nn.Linear(57, 60)
        self.fc2 = nn.Linear(117, 60)
        self.fc3 = nn.Linear(177, 2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        m1 = torch.cat([x, out], 1)
        out = F.relu(self.fc2(m1))
        out = torch.cat([m1, out], 1)
        out = F.relu(self.fc3(out))
        return out

    def name(self):
        return 'cross_mlpnet'

model = cross_MLPNet()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data
        print(pred)


for epoch in range(1, 50 + 1):
    train(epoch)
    test()