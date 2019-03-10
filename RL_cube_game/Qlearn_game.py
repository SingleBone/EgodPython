#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:16:19 2019

@author: egod
"""

from maze_env import Maze
from RL_brain import QL

def update():
    for episode in range(100):
        
        observation = env.reset()
        
        
        while True:
            env.render()
            
            action = RL.choose_action(observation)
            observation_,reward,done = env.step(action)
            
            RL.learn(observation,action,reward,observation_)
            
            observation = observation_
            
            if done:
                break
            
    print('game over')
    env.destroy()
    
if __name__=='__main__':
    env = Maze()
    RL = QL(action_space=list(range(env.n_actions)))
    
    env.after(100,update)
    env.mainloop()        