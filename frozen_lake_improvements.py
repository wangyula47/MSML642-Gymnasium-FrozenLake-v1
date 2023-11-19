import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def train_q_learning_agent(episodes, render=False, render_interval=100):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    # 1. Initialize Q-table with small random values to encourage exploration
    q = np.random.uniform(low=-0.01, high=0.01, size=(env.observation_space.n, env.action_space.n))

    # 2. Dynamic Learning Rate Parameters
    initial_learning_rate = 0.8
    min_learning_rate = 0.01
    learning_rate_decay = 0.85

    # 3. Discount Factor for Long-term Reward
    initial_discount_factor = 0.95
    min_discount_factor = 0.1
    discount_factor_decay = 0.99

    # 4. Exploration Parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # 5. Penalty for Falling into Holes
    hole_penalty = -1

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        terminated = False

        # Adjust learning rate and discount factor dynamically
        learning_rate = max(initial_learning_rate * np.power(learning_rate_decay, episode // 100), min_learning_rate)
        discount_factor = max(initial_discount_factor * np.power(discount_factor_decay, episode // 100), min_discount_factor)

        while not terminated:
            # Balancing exploration and exploitation
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, _, _ = env.step(action)

            # Apply hole penalty
            if terminated and reward == 0:
                reward = hole_penalty

            # Q-table update with long-term memory component
            q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state, :])
                                                                   - q[state, action])

            state = new_state
            total_reward += reward

        # Decaying exploration probability
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # 6. Efficient Rendering
        if render and episode % render_interval == 0:
            env.render()

    env.close()

    # Plotting cumulative rewards
    plt.plot(np.cumsum(rewards_per_episode))
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards over Episodes')
    plt.savefig('frozen_lake8x8_improved.png')

    # Saving Q-table
    with open("frozen_lake8x8_improved.pkl", "wb") as file:
        pickle.dump(q, file)


def test_agent(render=False):
    with open("frozen_lake8x8_q_table.pkl", "rb") as file:
        q = pickle.load(file)

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    total_reward = 0
    state = env.reset()[0]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.argmax(q[state, :])  # Greedy action selection
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()

    env.close()
    print("Total Reward: ", total_reward)


if __name__ == '__main__':
    train_q_learning_agent(500, render=True)
    test_agent(render=True)