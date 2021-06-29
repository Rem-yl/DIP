import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import torch.optim as optim
from torchvision import datasets, transforms ,models
from torch.autograd import Variable
from test_dataset import TestDateset

train_transform = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),

    # transforms.Resize(50),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_transforms = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    # transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.ColorJitter(saturation=5),
    # transforms.ColorJitter(saturation=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.ColorJitter(contrast=5),
    # transforms.ColorJitter(contrast=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])



# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
    transforms.Resize(50),
    transforms.CenterCrop(48),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

def dataload(batch_size,train_transform=data_rotate):



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