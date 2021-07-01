import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
import torch

lr = 0.001
epochs = 200000

#正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
def normfun(x,param):
    mu = param[0]
    sigma = param[1]
    result = torch.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * torch.sqrt(2*torch.tensor(np.pi,dtype=torch.float32)))

    return result


if __name__ == '__main__':
    x = [7,7.5,8,8.4,8.8,9.2,9.5,9.8]
    y = [2.636,2.638,2.591,2.494,2.292,1.908,1.591,1.301]

    x_tensor = torch.tensor(x,dtype=torch.float32,requires_grad=True)
    y_tensor = torch.tensor(y,dtype=torch.float32,requires_grad=True)

    loss_fn = torch.nn.MSELoss()

    # param = torch.tensor([10.,1.],dtype=torch.float32,requires_grad=True)
    param = Variable(torch.DoubleTensor([1.,1.]),requires_grad=True)
    # param = Variable(torch.randn(2,dtype=torch.float32),requires_grad=True)

    optimizer = torch.optim.SGD([param],lr=lr)

    for epoch in range(epochs):

        pred_result = normfun(x_tensor,param)
        optimizer.zero_grad()
        # print(pred_result)
        loss = loss_fn(pred_result,y_tensor).sum()

        loss.backward()
        optimizer.step()

        if epoch % 10000 == 0:
            print('epoch:{},param={},loss={}'.format(epoch,param,loss.item()))

    # print(x_tensor,y_tensor)

    # plt.plot(x,y,'bo',label='original')
    # plt.legend()
    # plt.show()
