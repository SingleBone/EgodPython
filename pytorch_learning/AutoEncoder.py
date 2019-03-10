import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 128
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# Load training data
train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
        )
print('train_data size :')
print(train_data.data.size())
print(train_data.targets.size())
train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        )
plt.imshow(train_data.data[0].numpy(),cmap='gray')
plt.title('%d'%train_data.targets[0])
plt.show()
plt.pause(0.5)

# define AutoEncoder class
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(28*28,256),
                nn.Tanh(),
                nn.Linear(256,64),
                nn.Tanh(),
                nn.Linear(64,16),
                nn.Tanh(),
                nn.Linear(16,4),
                )
        
        self.decoder = nn.Sequential(
                nn.Linear(4,16),
                nn.Tanh(),
                nn.Linear(16,64),
                nn.Tanh(),
                nn.Linear(64,256),
                nn.Tanh(),
                nn.Linear(256,28*28),
                nn.Sigmoid(),
                )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return (encoded,decoded)

# Initial
autoencoder = AutoEncoder().to('cuda')
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func = nn.MSELoss().to('cuda')

fig,ax = plt.subplots(2,N_TEST_IMG,figsize=(5,2))
plt.ion()

view_data = train_data.data[:N_TEST_IMG].view(-1,28*28).to(torch.FloatTensor())/255
for i in range(N_TEST_IMG):
    a = ax[0][i]
    a.imshow(np.reshape(view_data.data.numpy()[i],(28,28)),cmap='gray')
    a.set_xticks(());a.set_yticks(())

# Train Process
for epoch in range(EPOCH):
    for step,(x,labels) in enumerate(train_loader):
        b_x = x.view(-1,28*28).to('cuda')
        b_y = x.view(-1,28*28).to('cuda')
        
        _,decoded = autoencoder.to('cuda')(b_x)
        loss = loss_func(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step%100 == 0:
            
            _,decoded_data = autoencoder(view_data.to('cuda'))
            for i in range(N_TEST_IMG):
                a = ax[1][i]
                a.clear()
                a.imshow(decoded_data.to('cpu').view(-1,28,28).data.numpy()[i],cmap='gray')
                a.set_xticks(());a.set_yticks(())
            plt.draw();plt.pause(0.05)
            print('EPOCH : ',epoch+1,' | STEP : ',step+1,' | loss = %.4f'%loss.to('cpu').data.numpy())

plt.ioff()
plt.show()
        
