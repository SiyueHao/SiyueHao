import matplotlib
import numpy as np
import sys
import time
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

from envs import BlackjackEnv
import plotting
matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA  # Initialize the action probabilities
        best_action = np.argmax(Q[observation])      # Find the best action
        A[best_action] += (1.0 - epsilon)            # Add (1 - epsilon) probability to the best action
        return A
    return policy_fn

def mc_first_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance to sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities.
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用 int 更合理
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    #用于记录每一局的 reward
    all_rewards = []

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################
        
        
        # Step 1: Generate an episode: an array of (state, action, reward) tuples
        episode=[]
        state=env.reset()
        while True:
            probs=policy(state)
            action=np.random.choice(np.arange(len(probs)),p=probs)
            next_state,reward,done,_=env.step(action)
            episode.append((state,action,reward))
            
            if done:
                break
            state=next_state



        # Step 2: Find first-visit index for each (state, action) pair
        total_reward = sum([x[2] for x in episode])
        all_rewards.append(total_reward)
        G=0
        for t in range(len(episode)-1,-1,-1):
            state,action,reward=episode[t]
            # 更新累积回报 G = r + gamma * G_next
            G=discount_factor*G+reward
            has_visited_before = False
            for t2 in range(0,t):
                if episode[t2][0]==state and episode[t2][1]==action:
                    has_visited_before=True
                    break
            if not has_visited_before:# 如果是第一次出现，更新统计数据
                returns_sum[(state,action)]+=G
                returns_count[(state,action)]+=1
                # 更新 Q 值 = 总回报 / 次数
                Q[state][action]=returns_sum[(state,action)]/returns_count[(state,action)]

        # Step 3: Calculate returns backward, update only at first-visit time step
        #########################Implement your code end#########################
    return Q, policy, all_rewards


def mc_every_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用 int 更合理
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    all_rewards = []
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################
        
        # Step 1: Generate an episode
        episodes=[]
        state=env.reset()
        while True:
            probs=policy(state)
            action=np.random.choice(np.arange(len(probs)),p=probs)
            next_state,reward,done,_=env.step(action)
            episodes.append((state,action,reward))
            
            if done:
                break
            state=next_state

        
        # Step 2: Calculate returns for each (state, action) pair (every-visit)
        total_reward = sum([x[2] for x in episodes])
        all_rewards.append(total_reward)
        G=0
        for t in range(len(episodes)-1,-1,-1):
            state,action,reward=episodes[t]
            G=discount_factor*G+reward
            returns_sum[(state,action)]+=G
            returns_count[(state,action)]+=1
            Q[state][action]=returns_sum[(state,action)]/returns_count[(state,action)]
        
        #########################Implement your code end#########################

    return Q, policy, all_rewards

def plot_learning_curve(rewards_first, rewards_every, window=5000):
    """
    绘制滑动平均曲线
    window: 滑动窗口大小，越大曲线越平滑
    """
    # 计算滑动平均
    series_first = pd.Series(rewards_first).rolling(window=window).mean()
    series_every = pd.Series(rewards_every).rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    
    # 绘制曲线
    plt.plot(series_first, label="First-visit MC", color='blue', alpha=0.6)
    plt.plot(series_every, label="Every-visit MC", color='orange', alpha=0.6) 
    
    plt.title(f"Learning Curve Comparison (Moving Average window={window})")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_curve.png")
    plt.show()

if __name__ == "__main__":
    # First-Visit Monte Carlo 1e4 episodes
    Q_first, policy_first,reward_first = mc_first_visit(env, num_episodes=500000, epsilon=0.1)
    V = defaultdict(float)
    for state, actions in Q_first.items():
        V[state] = np.max(actions)
    plotting.plot_value_function(V, title="Optimal Value Function", 
        file_name="First_Visit_Value_Function_Episodes_500000")
    
    

    # First-Visit Monte Carlo 5e5 episodes
    #Q_500k, policy_500k = mc_first_visit(env, num_episodes=500000, epsilon=0.1)
    #V_500k = defaultdict(float)
    #for state, actions in Q_500k.items():
        #V_500k[state] = np.max(actions)
    #plotting.plot_value_function(V_500k, title="Optimal Value Function (500k Episodes)", 
                                 #file_name="Value_Function_500k")
    
    #Every-Visit Monte Carlo 5e5
    Q_every, policy_every,reward_every = mc_every_visit(env, num_episodes=500000, epsilon=0.1)
    V = defaultdict(float)
    for state, actions in Q_every.items():
         V[state] = np.max(actions)
    plotting.plot_value_function(V, title="Optimal Value Function", 
         file_name="Every_Visit_Value_Function_Episodes_500000")
    
    plot_learning_curve(reward_first, reward_every)