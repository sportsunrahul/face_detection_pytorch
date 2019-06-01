import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1, padding = 1)
        self.conv2 = nn.Conv2d(32,64,3,1, padding=1)
        self.conv3 = nn.Conv2d(64,128,3,1,padding=1)
        self.conv4 = nn.Conv2d(128,256,3,1,padding=1)
        self.fc1 = nn.Linear(7*7*256,4*4*128)
        self.fc2 = nn.Linear(4*4*128,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(4*4*128)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,7*7*256)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
