import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class MiniVGG_BN(nn.Module):
    def __init__(self, num_classes=6):

        super(MiniVGG_BN, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1_1   = nn.BatchNorm2d(16)

        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1_2   = nn.BatchNorm2d(16)

        # Block 2
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_1   = nn.BatchNorm2d(32)


        # Block 3
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(64)

        # Block 4
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4_1   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self,x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.pool(x)
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


