# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class OBJECTS:
    def __init__(self,LOC,COLOR):
        self.loc = LOC
        self.color = COLOR
        
class MYGAME:
    def __init__(self,STATES,OBJECTS):
        self.states = STATES # shape of states
        self.objects = OBJECTS # barriers or terminal
        self.action_space = ['up','down','left','right']
        self.n_actions = len(self.action_space) # num of actions
        self.title = 'GAME'
        self.bw = 100 # width of block
        self.env_rows = self.states[0]*self.bw # heigh of game window
        self.env_cols = self.states[1]*self.bw # width of game window
        self.eps = 0.9  # 1-eps is the probability of taking random aciton
        self.alpha = 0.2 # learing factor
        self.gamma = 0.9 # memery fade factor
        self.Max_Episodes = 100 # max try times
        self.N_objects = len(OBJECTS)
        self.s = (0,0)
        self._s = (0,0)
        self.s_ = (0,0)
        self.reward = 0
        self.out = None
        self.is_terminated = False
        self.build_Qtable()
        _,self.figure = plt.subplots(1,1,figsize=(5,5))
        self.initial()
    
    def initial(self):
        self.env = np.ones((self.env_rows,self.env_cols,3),dtype=np.float32)            
        for o in self.objects:
            r,c = o.loc
            self.env[r*self.bw:(r+1)*self.bw,c*self.bw:(c+1)*self.bw,:] = o.color
        self.episode = 0
        self.step_counter = 0
        self.s = (0,0)
        self._s = (0,0)
        self.s_ = (0,0)
        self.reward = 0
        self.out = None
        self.is_terminated = False
        self.action = None
        
    def build_Qtable(self):
        r,c = self.states
        idx = []
        for i in range(r):
            for j in range(c):
                idx.append((i,j))
        self.table = pd.DataFrame(
                np.zeros((r*c,self.n_actions)),
                index=idx,
                columns=self.action_space,
                )  
    
    def choose_action(self):
        r,c = self.s
        state_actions = self.table.loc[[(r,c)],:]
        ran = np.random.uniform()
        if (ran > self.eps) or (np.array(state_actions).all() == 0):
            action = np.random.choice(self.action_space)
        else:
            action = state_actions.idxmax(axis=1)[0]
        self.action = action
    
    def env_feedback(self):
        s = self.s
        N = self.N_objects
        objects = self.objects
        state_action = self.action
        if state_action == 'up':
            if s[0] == 0:
                self.s_ = s
                self.reward = 0
            else:
                self.s_ = (s[0]-1,s[1])
                self.reward = 0.0
                if s[0] == objects[-1].loc[0]+1 and s[1] == objects[-1].loc[1]:
                    self.out = 'alive'
                    self.is_terminated = True
                    self.reward = 1.0
                for i in range(N-1):
                    if s[0] == objects[i].loc[0]+1 and s[1] == objects[i].loc[1]:
                        self.out = 'death'
                        self.reward = -1.0
                        self.is_terminated = True
                        break
        elif state_action == 'down':
            if s[0] == self.states[0]-1:
                self.s_ = s
                self.reward = 0
            else:
                self.s_ = (s[0]+1,s[1])
                self.reward = 0.0
                if [0] == objects[-1].loc[0]-1 and s[1] == objects[-1].loc[1]:
                    self.out = 'alive'
                    self.reward = 1.0
                    self.is_terminated = True
                for i in range(N-1):
                    if s[0] == objects[i].loc[0]-1 and s[1] == objects[i].loc[1]:
                        self.out = 'death'
                        self.reward = -1.0
                        self.is_terminated = True
                        break
        elif state_action == 'left':
            if s[1] == 0:
                self.s_ = s
                self.reward = 0
            else:
                self.s_ = (s[0],s[1]-1)
                self.reward = 0.0
                if s[0] == objects[-1].loc[0] and s[1] == objects[-1].loc[1]+1:
                    self.out = 'alive'
                    self.reward = 1.0
                    self.is_terminated = True
                for i in range(N-1):
                    if s[0] == objects[i].loc[0] and s[1] == objects[i].loc[1]+1:
                        self.out = 'death'
                        self.reward = -1.0
                        self.is_terminated = True
                        break
        elif state_action == 'right':
            if s[1] == self.states[1]-1:
                self.s_ = s
                self.reward = 0
            else:
                self.s_ = (s[0],s[1]+1)
                self.reward = 0.0
                if s[0] == objects[-1].loc[0] and s[1] == objects[-1].loc[1]-1:
                    self.out = 'alive'
                    self.reward = 1.0
                    self.is_terminated = True
                for i in range(N-1):
                    if s[0] == objects[i].loc[0] and s[1] == objects[i].loc[1]-1:
                        self.out = 'death'
                        self.reward = -1.0
                        self.is_terminated = True
                        break
        
    def env_update(self):
        for o in self.objects :
            r,c = o.loc
            self.env[r*self.bw:(r+1)*self.bw,c*self.bw:(c+1)*self.bw,:] = o.color
        r,c = self.s
        self.env[r*self.bw:(r+1)*self.bw,c*self.bw:(c+1)*self.bw,:] = np.array([1,0,0],dtype=np.float32)
        if self.s != self._s:
            r,c = self._s
            self.env[r*self.bw:(r+1)*self.bw,c*self.bw:(c+1)*self.bw,:] = np.array([1,1,1],dtype=np.float32)
        
        self.figure.clear()
        self.figure.imshow(env.env)
        self.figure.set_xticks(());self.figure.set_yticks(())
        plt.pause(0.3)
	
        if self.is_terminated:
            print(chr(0x2605)*30)
            print('\a Episode : %s | Steps : %s | '%(self.episode,self.step_counter),self.out)
            print(chr(0x2605)*30)
            time.sleep(2)
            print('\f')
        
    def sayout(self,flag):
        if flag:
            print('\fin episode %s ,step %s :\n'%(self.episode+1,self.step_counter))
            print('state is %s ,direction is %s\n'%(self.s,self.action))
            print(self.table)
    
    def RL(self):
        self.build_Qtable()
        for episode in range(self.Max_Episodes):
            self.initial()
            self.env_update()
            self.episode = episode
            
            while not self.is_terminated:
                
                self.choose_action()
                self.env_feedback()
                r,c = self.s
                q_env = self.table.loc[[(r,c)],self.action]
                
                if not self.is_terminated:
                    r1,c1 = self.s_
                    q_target = self.reward + self.gamma*self.table.loc[[(r1,c1)],:].max(axis=1)[0]
                else:
                    q_target = self.reward 

                self.table.loc[[(r,c)],self.action] += self.alpha*(q_target-q_env)
                self.sayout(0)
                self._s = self.s
                self.s = self.s_
                self.env_update()
                self.step_counter += 1
                
if __name__ == '__main__':
    STATES = np.array([4,4])
    objects = [OBJECTS((1,2),(0,0,0)),OBJECTS((2,1),(0,0,0)),OBJECTS((2,2),(0,1,0))]
    env = MYGAME(STATES,objects)
    env.RL()
