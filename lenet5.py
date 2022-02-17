import torch
import torch.nn as nn

class lenet5(nn.Module):

    def __init__(self):
        super(lenet5, self).__init__()

        # set-1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2, bias=False)
        self.bin1 = nn.ReLU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # set-2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False)
        self.bin2 = nn.ReLU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # set-3
        self.fc3 = nn.Linear(50*7*7, 500, bias=False)
        self.bin3 = nn.ReLU()

        # set-4
        self.fc4 = nn.Linear(500, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bin1(x)

        x = self.avg_pool1(x)

        x = self.conv2(x)
        x = self.bin2(x)

        x = self.avg_pool2(x)

        x = x.view(-1, 7*7*50)
        x = self.fc3(x)
        x = self.bin3(x)

        x = self.fc4(x)

        return x