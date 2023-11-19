import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def initialize_environment():
    # Registering the custom Frozen Lake environment
    gym.register(
        id="FrozenLakeEnhanced-v0",
        entry_point="frozen_lake_enhanced:FrozenLakeEnv",
        kwargs={"map_name": "8x8"},
        max_episode_steps=200,
        reward_threshold=0.85
    )


def train_agent(episodes, render_mode='human'):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode=render_mode)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99995

    cumulative_rewards = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if render_mode == 'human':
                env.set_q(q_table)
                env.set_episode(episode)

            # Q-learning update
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state, action] = new_value

            state = next_state

        # Epsilon decay
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        cumulative_rewards[episode] = cumulative_rewards[episode - 1] + (1 if reward == 1 else 0)

    env.close()

    # Save the trained model
    with open('frozen_lake8x8_q_table.pkl', 'wb') as file:
        pickle.dump(q_table, file)

    # Plotting the learning curve
    plt.plot(cumulative_rewards)
    plt.title("Cumulative Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.savefig('frozen_lake8x8_training.png')


if __name__ == '__main__':
    initialize_environment()
    train_agent(20, render_mode='human')
