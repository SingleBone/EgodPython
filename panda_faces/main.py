# packages
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.sparse as ss

# Hyper Parameters
Epoch = 20000
DataSize = 364
BatchSize = 256
GenLR = 0.00005 # Learning Rate of Generator
DesLR = 0.00005 # Learning Rate of Descriminator
DataSet = []
ImgSize = (1<<6,1<<6)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loading
TransComp = transforms.Compose([
        transforms.ToTensor()
        ])
root = './images/'
for filename in os.listdir(root):
    img = cv2.imread(root+filename,0)
    img = cv2.resize(img,ImgSize)
    img = np.array(img)    
    DataSet.append(img)
DataSet = np.array(DataSet) 
DataSet = TransComp(DataSet).permute(1,2,0).float()
class Descriminator(nn.Module):
    def __init__(self):
        super(Descriminator,self).__init__()
        self.f = nn.Sequential(
                nn.Conv2d(1,8,5,1,2),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(8,16,5,1,2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,32,5,1,2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )       
        self.out = nn.Sequential(
                nn.Linear(32*4*4,1),
                nn.Sigmoid(),
                )
    def forward(self,x):
        x = x.unsqueeze(1).float()
        x = self.f(x)
        x = x.view(-1,32*4*4)
        x = self.out(x)
        return x
'''        

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,8,5,1,2),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(8,16,5,1,2),
                nn.BatchNorm2d(16),
                nn.ReLU()
                )
        self.out = nn.Sequential(
                nn.Linear(16*40*40,1*40*40),
                )
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,16*40*40)
        x = self.out(x)
        x = x.view(-1,40,40)
        return x
'''

        
if __name__ == '__main__':
    
    if not os.path.exists('./panda_draw'):
        print('first try, creating a model...')
        G = nn.Sequential(
                nn.Linear(1<<6,1<<8),
                nn.ReLU(),
                nn.Linear(1<<8,1<<10),
                nn.Sigmoid(),
                nn.Linear(1<<10,1<<12),
                nn.Sigmoid(),
                )
        D = Descriminator()
    
        G_opt = torch.optim.Adam(G.parameters(),lr=GenLR)
        D_opt = torch.optim.Adam(D.parameters(),lr=DesLR)
        epoch = 0
    else:
        checkpoint = torch.load('./panda_draw')
        G = checkpoint['G']
        D = checkpoint['D']
        G_opt = checkpoint['G_opt']
        D_opt = checkpoint['D_opt']
        epoch = checkpoint['epoch']
        print('already train %s epoch..'%epoch)
    
    
    
    G = G.cuda()
    D = D.cuda()
    
    
    for step in range(epoch+1,epoch+Epoch+1):
        index = np.random.permutation(364)[:BatchSize]
        pandas = DataSet[index]
        
        idea = np.ones((BatchSize,1<<6))
        idx = np.random.permutation(1<<6)[:1<<4]
        idea[idx] = 0
        idea = torch.from_numpy(idea)
        idea = idea.float()
        
        GenPandas = G(idea.cuda()).view(-1,1<<6,1<<6).float()
        
        IsPandasR = D(pandas.cuda())
        IsPandasG = D(GenPandas)
        
        D_loss = -torch.mean(torch.log(IsPandasR) + torch.log(1.-IsPandasG))
        G_loss = torch.mean(torch.log(1.-IsPandasG))
        
        G_opt.zero_grad()
        D_opt.zero_grad()
        D_loss.backward(retain_graph = True)
        G_loss.backward()
        D_opt.step()
        G_opt.step()
        
        if step% 100 == 0:
            print('Step : %s, D_loss: %.4f, G_loss: %.4f'%(step+epoch+1,D_loss,G_loss))
            plt.imshow(GenPandas.view(-1,1<<6,1<<6)[0].cpu().data.numpy(),cmap = 'gray')
            plt.show()
        
            torch.save({
                        'G':G,
                        'D':D,
                        'G_opt':G_opt,
                        'D_opt':D_opt,
                        'epoch':step
                    },'./panda_draw')
                
