import torch

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