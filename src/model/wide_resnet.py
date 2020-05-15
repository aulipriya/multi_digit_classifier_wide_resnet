# From https://github.com/xternalz/WideResNet-pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as fn


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = fn.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, drop_rate=0.0):
        super(WideResNet, self).__init__()
        num_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, num_channels[0], num_channels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, num_channels[1], num_channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, num_channels[2], num_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = num_channels[3]

        self._digit_length = nn.Linear(num_channels[3], 7)
        self._digit1 = nn.Linear(num_channels[3], 11)
        self._digit2 = nn.Linear(num_channels[3], 11)
        self._digit3 = nn.Linear(num_channels[3], 11)
        self._digit4 = nn.Linear(num_channels[3], 11)
        self._digit5 = nn.Linear(num_channels[3], 11)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = fn.avg_pool2d(out, 8)

        out = out.view(-1, self.nChannels)
        length_logits = self._digit_length(out)
        digit1_logits = self._digit1(out)
        digit2_logits = self._digit2(out)
        digit3_logits = self._digit3(out)
        digit4_logits = self._digit4(out)
        digit5_logits = self._digit5(out)

        return length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits


def test_model():
    cnn = WideResNet(depth=16, widen_factor=8,
                     drop_rate=0.4)

    input_tensor = torch.rand((2, 3, 54, 54))
    length, digit1, digit2, digit3, digit4, digit5 = cnn(input_tensor)
    print(length.shape)
    print(digit1.shape)
    print(digit2.shape)
    print(digit3.shape)
    print(digit4.shape)
    print(digit5.shape)


if __name__ == '__main__':
    test_model()


