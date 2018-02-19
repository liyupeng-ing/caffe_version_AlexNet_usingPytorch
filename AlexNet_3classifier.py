# -*- coding: utf-8 -*-

import myAlexNet as models
my_alexnet = models.alexnet(pretrained=True)
print(my_alexnet.conv2.weight)
