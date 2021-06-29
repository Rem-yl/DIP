import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import torch.optim as optim
from torchvision import datasets, transforms ,models
from torch.autograd import Variable

# input_size = 48
class Net(nn.Module):
    def __init__(self,nclasses=43,num_fc=1184):
        super(Net, self).__init__()
        self.num_fc = num_fc

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(num_fc, 300)
        self.fc2 = nn.Linear(300, nclasses)
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(29)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.dense1_bn = nn.BatchNorm1d(300)

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4,32),
            nn.ReLU(True),
            nn.Linear(32,3*2)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

    def backbone(self,x):

        feature = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        feature = F.relu(self.conv2_bn(self.conv2(feature)))
        feature = F.relu(self.conv3_bn(self.conv3( self.maxpool2(feature))))
        feature = F.relu(self.conv4_bn(self.conv4( self.maxpool3(feature))))
        feature = self.maxpool4(feature) 

        return feature

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)

        theta = self.fc_loc(xs)


        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def fc_layer(self,x):
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)

        return F.log_softmax(x,dim=1)


    def forward(self, x):
        # x = self.stn(x)
        # print(x.size())

        feature = self.backbone(x)
        # print(feature.shape)

        feature = feature.view(-1,self.num_fc)
        out = self.fc_layer(feature)

        return out




        # x = F.relu(self.fc1(x))
        # x = self.dense1_bn(x)
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # x = F.dropout(x, training=self.training)
        # return F.log_softmax(x,dim=1)

