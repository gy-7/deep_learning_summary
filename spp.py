import torch
from torch import nn
import torch.nn.functional as F


class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.silu(x)
        return x


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):  # 这里5，9，13，就是初始化的kernel size
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = CBS(in_channels, mid_channels, 1, 1)
        self.conv2 = CBS(mid_channels * (len(k) + 1), out_channels, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = torch.cat([inputs] + [pool(inputs) for pool in self.pools], 1)
        inputs = self.conv2(inputs)
        return inputs


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pools_num=3):
        super().__init__()
        self.pools_num = pools_num
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(self.pools_num)])
        self.conv3 = CBS(in_channels * (pools_num + 1), out_channels, 1, 1, 0)

    def forward(self, inputs):
        outputs = [inputs, ]
        for i in range(self.pools_num):
            outputs.append(self.pools[i](outputs[i]))

        outputs = torch.cat(outputs, 1)
        outputs = self.conv3(outputs)
        return outputs


class SPPF_Down(nn.Module):
    def __init__(self, in_channels, out_channels, pools_num=3):
        super().__init__()
        self.pools_num = pools_num
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(self.pools_num)])
        self.conv3 = CBS(in_channels * (pools_num + 1), out_channels, 3, 2, 1)

    def forward(self, inputs):
        outputs = [inputs, ]
        for i in range(self.pools_num):
            outputs.append(self.pools[i](outputs[i]))

        outputs = torch.cat(outputs, 1)
        outputs = self.conv3(outputs)
        return outputs


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)

    net = SPP(3, 5)
    outputs0 = net(inputs)
    print(outputs0.shape)

    net1 = SPPF(3, 5, 6)
    outputs1 = net1(inputs)
    print(outputs1.shape)

    net2 = SPPF_Down(3, 5, 6)
    outputs2 = net2(inputs)
    print(outputs2.shape)
