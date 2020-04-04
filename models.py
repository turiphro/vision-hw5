import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self, lr):
        return optim.SGD(self.parameters(),
                         lr=lr,
                         momentum=0.9, weight_decay=5e-4)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr * (0.5 ** (epoch // 30))

        if epoch % 30 == 0:
            print("Learning rate is now", lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def num_flat_features(self, x):
        size = x.size()[1:] # exclude batch dimension
        n = 1
        for s in size:
            n *= s
        return n


class LazyNet(BaseModel):
    # Training on turiphro's laptop: 34% acc, 0h31m
    def __init__(self):
        super(LazyNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x


class BoringNet(BaseModel):
    # Training on turiphro, cpu,  batch=4:  49% acc, 0h47m
    # Training on turiphro, cpu,  batch=64: 45% acc, 0h15m
    # Training on turiphro, cuda, batch=4:  48% acc, 0h40m
    # Training on turiphro, cuda, batch=64: 51% acc, 0h11m
    def __init__(self):
        super(BoringNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class CoolNet(BaseModel):
    # Training on turiphro's laptop, chan=[32,64], batch=64: 70% acc, 2h37m
    # Training on turiphro's laptop, chan=[64,128], batch=16: 74% acc, 10h16m
    def __init__(self, channels=[64, 128]):
        super(CoolNet, self).__init__()

        # based on VGG (first two blocks)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(channels[1] * 22 * 22, 512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SuperCoolNet(BaseModel):
    # Training on turiphro, cpu,  batch=64: 80% acc, 7h00m
    # Training on turiphro, cuda, batch=64: 79% acc, 2h21m

    def __init__(self):
        super(SuperCoolNet, self).__init__()

        # based on VGG11 (first 5 blocks)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x
