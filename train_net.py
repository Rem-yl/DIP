from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import torch.optim as optim
from torchvision import datasets, transforms ,models
from torch.autograd import Variable

from dataload import dataload
from network import Net
import config 

import logging
import pandas as pd

# 网络超参数
batch_size = 50                                                                                                                                                                                     
epochs = 1000                                                                                                                                                                                                                                                                                                                                                                                
seed = 1                                                                                                                                                                                           
log_interval=180                                                                                                                                                                                   
data = "data"                                                                                                                                                                                                                                                                                                                                                                                          
torch.manual_seed(1)   
batch_size = 16
validation_split = .2
shuffle_dataset = True

lr = 0.007                                                                                                                                                                                          
momentum = 0.8                                                                                                                                                                                     
decay = 0.9                                                                                                                                                                                        
step = 1000                                                                                                                                                                                        
l2_norm = 0.00001  
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
resume = False
device = 0
# These may change as described in paper

nclasses = 43 # GTSRB as 43 classes

filename = r'/opt/mnt/yl/DIP/project/code/log.txt'

config_dict = {
    'batch_size':batch_size,
    'epochs':epochs,
    'batch_size':batch_size,
    'lr':lr,
    'momentum':momentum,
    'decay':decay,
    'step':step,
    'l2_norm':l2_norm,
    'cuda':cuda,
    'device':device
}



def train(model,epoch,train_loader):
    model.train()
    
    logging.basicConfig(level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=filename)
    
    train_logger = logging.getLogger('Train')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target).cuda()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # print('Train Epoch: {},Loss:{}'.format(epoch,))
            train_logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0

    logging.basicConfig(level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=filename)

    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_loss /= len(val_loader.dataset)


    
    val_logger = logging.getLogger('Validation')
    val_logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    print('\n Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss

def test_acc():
    test_path = r'/opt/mnt/yl/DIP/dataset/GTSRB1/test'
    gt_path = r'/opt/mnt/yl/DIP/dataset/GTSRB1/GT.csv'
    gt = data = pd.read_csv(gt_path)
    label = data['ClassId'].values
    print(label)


# 读取数据
train_loader,val_loader = dataload(batch_size)

# 构建模型
model = Net()
print(device)
if  cuda: 
    model.to(device)

# test_acc()

optimizer = torch.optim.SGD(model.parameters(), lr=lr , momentum=momentum, weight_decay=l2_norm, nesterov=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

logging.basicConfig(level = logging.INFO,
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
filename=filename)

logging.basicConfig(level = logging.INFO,
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
filename=filename)
config_logger = logging.getLogger('Config')
config_logger.info('-'*30)
config_logger.info('Start Logger ...')
config_logger.info(config_dict)
config_logger.info(model)

# logger = logging.getLogger('Config')
# # logger.info('Start Logger')
# logging.info(model)
# logging.info(config)


temp = 10
for epoch in range(1, epochs):
    train(model,epoch, train_loader)
    val = validation()
    if epoch % step :
        scheduler.step()
    if epoch % temp == 0:
        # 每隔10个epcoh保存模型
        torch.save(model,'/opt/mnt/yl/DIP/project/model/model_{}.pth'.format(epoch))


