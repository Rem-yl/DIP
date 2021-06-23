import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import torch.optim as optim
from torchvision import datasets, transforms ,models
from torch.autograd import Variable
from test_dataset import TestDateset



def dataload(batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(50),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    csv_path = r'/opt/mnt/yl/DIP/dataset/GTSRB1/GT.csv'
    test_path = r'/opt/mnt/yl/DIP/dataset/GTSRB1/test'

    train_data_path = '/opt/mnt/yl/DIP/dataset/GTSRB1/train'

    train_dataset = datasets.ImageFolder(train_data_path,transform=train_transform)
    test_dataset = TestDateset(csv_path,test_path,transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=1)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=True, num_workers=1)

    # dataset = datasets.ImageFolder(train_data_path,transform=train_transform)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size, shuffle=True, num_workers=1)


    # val_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader,test_loader