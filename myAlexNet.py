# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1    = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.relu1    = nn.ReLU(inplace=True)
        self.pool1    = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2    = nn.Conv2d(96, 256, kernel_size=5, padding=2,groups=2)
        self.relu2    = nn.ReLU(inplace=True)
        self.pool2    = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3    = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3    = nn.ReLU(inplace=True)
        self.conv4    = nn.Conv2d(384, 384, kernel_size=3, padding=1,groups=2)
        self.relu4    = nn.ReLU(inplace=True)
        self.conv5    = nn.Conv2d(384, 256, kernel_size=3, padding=1,groups=2)
        self.relu5    = nn.ReLU(inplace=True)
        self.pool5    = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop6    = nn.Dropout()
        self.fc6      = nn.Linear(256 * 6 * 6, 4096)
        self.relu6    = nn.ReLU(inplace=True)
        self.drop7    = nn.Dropout()
        self.fc7      = nn.Linear(4096, 4096)
        self.relu7    = nn.ReLU(inplace=True)
        self.fc8      = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.drop6(x)
        x = x.view(x.size(0),256 * 6 * 6)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop7(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)
        return x


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load('C:/Users/lypeng/AnacondaProjects/caffemodel/bvlc_reference_caffenet.caffemodel.pth'))
    return model
