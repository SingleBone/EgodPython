# Packeges
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import gym
import time

# Hyper Parameters
BATCH_SIZE = 5000
LR = 0.05
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 1
MEMORY_CAPACITY = 10000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# Basic Net
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.f1 = nn.Linear(N_STATES,50)
        self.f1.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(50,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)
        
    def forward(self,x):
        x = self.f1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# Define DQN class
class DQN(object):
    def __init__(self): # Bulid Target Net & Eval Net & Memory
        self.eval_net, self.target_net = Net().to(device),Net().to(device)
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY,N_STATES*2+2)) # Initialize the Memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr = LR)
        self.loss_func = nn.MSELoss()
        
    def choose_action(self,x): # choose aciton from observation
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        
        if np.random.uniform() < EPSILON:
            actions = self.eval_net.forward(x.to(device))
            action = torch.max(actions,1)[1].to('cpu').data.numpy()[0] # sort at the axis-1 and get the index
        else:
            action = np.random.randint(0,N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    
    def store_transition(self,s,a,r,s_): # store the memory
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter
        self.memory[index,:]=transition
        self.memory_counter += 1
    
    def learn(self): # update the Target Net by learning the memory
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: # update first
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        # sample a batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory = self.memory[sample_index,:]
        b_s = torch.FloatTensor(b_memory[:,:N_STATES]).view(-1,N_STATES)
        b_a = torch.LongTensor(b_memory[:,N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:,N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:,-N_ACTIONS:]).view(-1,N_STATES)
        
        # forming the q_target and q_eval
        q_eval = self.eval_net(b_s.to(device)).gather(1,b_a.to(device)) # shape(batch,1)
        q_next = self.target_net(b_s_.to(device)).detach()
        q_target = b_r.to(device) + GAMMA*q_next.max(1)[0]
        loss = self.loss_func(q_eval,q_target)
        
        # Calculation & update eval_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        


if __name__ == '__main__':
    device = 'cuda'
    dqn = DQN()
    for episode in range(500):
        s = env.reset()
        ep_r = 0    
        while True:
            since = time.time()
            env.render()
            a = dqn.choose_action(s)
            
            s_,r,done,info = env.step(a)
            
            x,x_dot,theta,theta_dot = s_
            
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2
            
            dqn.store_transition(s,a,r,s_)
            ep_r += r
            
            if dqn.memory_counter > MEMORY_CAPACITY-1:
                dqn.memory_counter = 0
                dqn.learn()

            if done:
                if episode%100 == 0:
                    to = time.time()
                    print('Ep: ', episode,'| time: %.4f'%(to - since),'s')
                    print('reward: %.4f | step %s | memory counter: %s'%(\
                            r,dqn.learn_step_counter,dqn.memory_counter))
                break
            
            s = s_
            
        
    env.close()
