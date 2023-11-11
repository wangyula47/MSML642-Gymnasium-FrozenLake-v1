#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install gym


# In[5]:


cd ~/Desktop


# In[6]:


mkdir FrozenLakeQLearning


# In[7]:


cd FrozenLakeQLearning


# In[9]:


import gym

def explore_frozenlake():
    # FrozenLake environment
    env = gym.make('FrozenLake-v1')

    # basic information about the environment
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    # reward structure
    print("Reward Table:")
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            for prob, next_state, reward, _ in transitions:
                print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")

if __name__ == "__main__":
    explore_frozenlake()


# In[10]:


cd ~/Desktop/FrozenLakeQLearning


# Observation Space: Discrete(16)
# 
# This means there are 16 possible states in the environment, numbered from 0 to 15. Each state represents a different position on the frozen lake.
# 
# Action Space: Discrete(4)
# 
# There are four possible actions the agent can take at each state. The actions are represented by integers 0, 1, 2, and 3.Reward Table:
# 
# The reward table shows the consequences of taking a particular action in a particular state.
# For example, when the agent is in state 0 and takes action 0, it stays in the same state (Next State: 0) with a reward of 0.0.
# When the agent is in state 0 and takes action 1, it moves to state 4 (Next State: 4) with a reward of 0.0.
# This pattern repeats for all combinations of states and actions.
# In reinforcement learning, the agent's goal is to learn a policy that maximizes the cumulative reward over time. The numbers in the reward table will be used by the Q-learning algorithm to determine the best actions to take in each state.
# 

# Q- Learning Algorithm and Training agent to navigate the frozenlake 

# In[ ]:




