#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd


# In[1]:


class RL():
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.action_space = action_space # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.table = pd.DataFrame(columns=self.action_space)
        self.terminated = False
    # state, observation, _s, s, s_ 都是一个状态值(同一种数据类型-元组)，可以是但不限于是一个坐标
    
    def check_state_exist(self,state):
        # 如果state不在表的索引中，则自动添加对应格式的一行
        if str(state) not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0]*len(self.action_space),
                    index = self.table.columns,
                    name = str(state),
                )
            )
    
    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_actions = self.table.loc[str(observation),:]
            # 打乱state_actions中的排序(索引和值仍然保持一开始的对应关系)
            # 这是为了防止state_actions中值都相同时总是作出一样的选择
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            action = np.random.choice(self.action_space)
        
        return action
            
    def learn(): # 学习，或者说更新表中值
        pass
        


# In[65]:


class QL(RL):
    
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(QL,self).__init__(action_space,learning_rate,reward_decay,e_greedy)
        
    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_pre = self.table.loc[str(s),a]
        if not self.terminated:
            q_tar = r + self.gamma*self.table.loc[str(s_),:].max() # 好处驱使，会表现的更想要奖励，甚至不畏死亡
        else: 
            q_tar = r
        self.table.loc[str(s),a] += self.lr*(q_tar-q_pre)


# In[ ]:


class SarsaL(RL):
    
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaL, self).__init__(action_space, learning_rate, reward_decay, e_greedy)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_pre = self.table.loc[str(s),a]
        if not self.terminated:
            q_tar = r + self.gamma*self.table.loc[str(s_),a_] # 对下一步记忆更深刻，会表现的更怕惩罚
        else: 
            q_tar = r
        self.table.loc[str(s),a] += self.lr*(q_tar-q_pre)


# In[5]:


class SarsaLambdaL(RL):
    
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
        super(SarsaLambdaL,self).__init__(action_space,learning_rate,reward_decay,e_greedy)
        
        self.lambda_ = trace_decay
        self.eligibility_trace = self.table.copy()
    
    def check_state_exist(self,state):
        if str(state) not in self.table.index:
            to_be_appended = pd.Series(
                    [0]*len(self.action_space),
                    index = self.table.columns,
                    name = str(state),
                )
            self.table = self.table.append(to_be_appended)
            self.eligibility_trace = self.eligibility_trace.append(to_be_appended)
            
    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_pre = self.table.loc[str(s),a]
        if not self.terminated :
            q_tar = r + self.gamma*self.table.loc[str(s_),a_]
        else:
            q_tar = r
        diff = q_tar - q_pre
        
        self.eligibility_trace.loc[str(s),:] *= 0
        self.eligibility_trace.loc[str(s),a] += 1
        
        self.table += self.lr*diff*self.eligibility_trace
        
        self.eligibility_trace *= self.gamma*self.lambda_
        


# In[ ]:




