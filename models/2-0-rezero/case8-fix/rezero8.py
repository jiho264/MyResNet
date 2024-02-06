"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


import torch
import sys, os, tqdm, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    Compose,
    RandomCrop,
    RandomShortestSize,
    AutoAugment,
    Normalize,
    CenterCrop,
    ToImage,
    ToDtype,
)
from torchvision import datasets

from src.Mymodel import MyResNet_CIFAR

BATCHE_SIZE = 128
NUM_EPOCHS = 180

train_dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="../../../data",
        train=True,
        download=False,
        transform=Compose(
            [
                ToImage(),
                ToDtype(torch.float32),
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    ),
    batch_size=BATCHE_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)
test_dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="../../../data",
        train=False,
        download=False,
        transform=Compose(
            [
                ToImage(),
                ToDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    ),
    batch_size=BATCHE_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)

criterion = torch.nn.CrossEntropyLoss()
# model = MyResNet_CIFAR(num_classes=10, num_layer_factor=5, Downsample_option="A").to(
#     "cuda"
# )
model = resnet32().to("cuda")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[82, 123], gamma=0.1
)

for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        outputs = model(inputs)  # A
        loss = criterion(outputs, labels)  # B

        optimizer.zero_grad()  # C
        loss.backward()  # D
        optimizer.step()  # E

        train_loss += loss.item()  # F
        _, predicted = outputs.max(1)  # G
        train_total += labels.size(0)  # H
        train_correct += predicted.eq(labels).sum().item()  # I

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)  # A
            _, predicted = outputs.max(1)  # B
            test_total += labels.size(0)  # C
            test_correct += predicted.eq(labels).sum().item()  # D
            test_loss += criterion(outputs, labels).item()  # E

    scheduler.step()
    print(
        f"Epoch {epoch} | Train Loss : {train_loss / len(train_dataloader):.4f} | Train Acc : {100 * train_correct / train_total:.2f}% | Test Loss : {test_loss / len(test_dataloader):.4f} | Test Acc : {100 * test_correct / test_total:.2f}%"
    )
