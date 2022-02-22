import torch.nn as nn
import torch.nn.functional as F


class xxNet(nn.Module):
    def __init__(self):
        super(xxNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(32, 16, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(16, 1, (3, 3), padding=1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


if __name__ == '__main__':
    print(xxNet())