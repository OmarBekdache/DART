import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Parallel BN ----------------
class _ParallelBN(nn.Module):
    """Holds two BatchNorm2d modules; shares affine params; separate running stats."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.main = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.aux  = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

        if affine:
            # Share the SAME parameters for gamma/beta so we don't double learnables
            self.aux.weight = self.main.weight
            self.aux.bias   = self.main.bias

        self.use_aux = False  # flipped externally

    def forward(self, x):
        return self.aux(x) if self.use_aux else self.main(x)

def set_parallel_bn_use_aux(module: nn.Module, use_aux: bool):
    """Recursively set which BN stats to use (main vs aux)."""
    for m in module.modules():
        if isinstance(m, _ParallelBN):
            m.use_aux = use_aux

# ---------------- ResNet blocks (use ParallelBN) ----------------
class BasicBlock_parallel_bn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = _ParallelBN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = _ParallelBN(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                _ParallelBN(self.expansion*planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_parallel_bn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = _ParallelBN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = _ParallelBN(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = _ParallelBN(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                _ParallelBN(self.expansion*planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_parallel_bn(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = _ParallelBN(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_parallel_bn(num_classes=10):
    return ResNet_parallel_bn(BasicBlock_parallel_bn, [2, 2, 2, 2], num_classes)

def ResNet34_parallel_bn(num_classes=10):
    return ResNet_parallel_bn(BasicBlock_parallel_bn, [3, 4, 6, 3], num_classes)

def ResNet50_parallel_bn(num_classes=10):
    return ResNet_parallel_bn(Bottleneck_parallel_bn, [3, 4, 6, 3], num_classes)

def ResNet101_parallel_bn(num_classes=10):
    return ResNet_parallel_bn(Bottleneck_parallel_bn, [3, 4, 23, 3], num_classes)

def ResNet152_parallel_bn(num_classes=10):
    return ResNet_parallel_bn(Bottleneck_parallel_bn, [3, 8, 36, 3], num_classes)