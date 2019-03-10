import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import time

# 超参数/全局参数

N_STATES = 10 #世界长度/状态个数
ACTIONS = ['left','right'] 
EPSILON = 0.9 # 贪婪度
ALPHA = 0.1 # 学习率
GAMMA = 0.9 # 奖励递减/衰退系数
MAX_EPISODES = 13 # 最大回合数
FRESH_TIME = 0.3 # 回合间隔时间

# Q表生成函数

def build_Q_table(n_states,actions):
    table = DF(np.zeros((n_states,len(actions))),columns=actions) 
    return table

# 行为选择函数

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:] #检索出状态为state时的动作倾向直
    choice = np.random.uniform() #随机选择直
    if (choice > EPSILON) or (state_actions.all() == 0): # 当随即选择直 > 贪婪度 或者 第一次进入该state时，随机选择行动
        action = np.random.choice(ACTIONS)
    else: # 否则按照动作倾向直选择行动
        action = state_actions.idxmax()
    return action

# 环境反馈函数

def env_feedback(state,action):
    if action == 'right': # 如果行动为向右
        if state == N_STATES-2: # 如果紧贴终点，则胜利并获得奖励
            state_pre = 'terminal'
            reward = 1
        else: # 否则向右走一步，无奖励
            state_pre = state + 1
            reward = 0
    else: # 如果向左
        reward=0 # 向左没有奖励
        if state == 0: # 如果已经靠墙，则原地踏步
            state_pre = state
        else: # 否则向左走一步
            state_pre = state - 1
    return state_pre,reward

# 环境更新函数

def update_env(state,episode,step_counter):
    env_list = ['-']*(N_STATES-1)+['T'] # 环境 = ------T
    if state == 'terminal': # 如果已经达到终点， 宣告回合结束并展示统计信息
        interaction = 'Episode : %s Total steps : %s'%(episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r           ',end='')
    else: # 否则，在冒险者所在位置 用 o 代替 - 表示其位置
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

# 强化学习主循环

def RL():
    q_table = build_Q_table(N_STATES,ACTIONS) # 初始化Q表
    
    for episode in range(MAX_EPISODES): #循环 MAX_EPISODES 回合
        step_counter = 0 # 初始步数
        state = 0 # 冒险者初始位置为最左边
        is_terminated = False # 初始胜利flag
        update_env(state,episode,step_counter) # 初始环境
        
        while not is_terminated: # 未触发胜利flag时，游戏一直循环进行
            action = choose_action(state,q_table) # 选择动作
            state_fb,reward = env_feedback(state,action) # 获得环境反馈：该state下采取该action后的下一步state_fb 和 奖励 reward
            q_env = q_table.loc[state,action] # 当前该state下采取action的q_env直
           
            if state_fb != 'terminal': # 如果 state_fb 不是终点 
                q_target = GAMMA*q_table.iloc[state_fb,:].max() # jiyizhongxiayibu获得的q直
            else: # 否则
                q_target = reward 
                is_terminated = True
        
            q_table.loc[state,action] += ALPHA*(q_target-q_env) # 更新 q_table
            state = state_fb # 更新state
        
            update_env(state,episode,step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = RL()
    print('\r\nQ-table:\n')
    print(q_table)

