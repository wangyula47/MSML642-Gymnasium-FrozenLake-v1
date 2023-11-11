import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def train_q_learning_agent(episodes, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table

    learning_rate = 0.9  # learning rate
    discount_factor = 0.9  # discount rate
    epsilon = 1.0  # Exploration probability
    epsilon_decay_rate = 0.0001  # Epsilon decay rate

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Random exploration
            else:
                action = np.argmax(q[state, :])  # Exploitation

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-value update only during training
            if epsilon > 0:
                q[state, action] += learning_rate * (
                        reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state
            total_reward += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        rewards_per_episode[episode] = total_reward

    env.close()

    # Plot cumulative rewards
    sum_rewards = np.cumsum(rewards_per_episode)
    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards over Episodes')
    plt.savefig('frozen_lake8x8.png')

    # Save the learned Q-table
    with open("frozen_lake8x8.pkl", "wb") as file:
        pickle.dump(q, file)


if __name__ == '__main__':
    train_q_learning_agent(1000, render=True)
