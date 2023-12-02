import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import pickle
from frozen_lake_enhanced import FrozenLakeEnhancedEnv

# Register the enhanced frozen lake environment
gym.envs.registration.register(
    id="FrozenLake-enhanced-v0",
    entry_point="frozen_lake_enhanced:FrozenLakeEnhancedEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,
)


def adjust_epsilon_decay(episode_rewards, current_decay, threshold=0.1, adjustment_factor=0.999):
    """
    Adjusts the epsilon decay rate based on the recent performance of the agent.

    :param episode_rewards: List of rewards from previous episodes.
    :param current_decay: Current epsilon decay rate.
    :param threshold: Performance improvement threshold to trigger decay adjustment.
    :param adjustment_factor: Factor to adjust the decay rate.
    :return: Adjusted epsilon decay rate.
    """
    if len(episode_rewards) > 10:  # Check performance over last 10 episodes
        recent_average = np.mean(episode_rewards[-10:])
        overall_average = np.mean(episode_rewards)
        if recent_average > overall_average * (1 + threshold):
            return max(current_decay * adjustment_factor, 0.01)  # Avoid too low decay rate
    return current_decay


# Function to run episodes (training or testing)
def run_episodes(env, q_table, episodes, learning_rate, discount_factor, epsilon, epsilon_decay, is_training, render=False,
                 track_outcomes=False):
    episode_rewards = np.zeros(episodes)
    current_epsilon_decay = epsilon_decay
    outcomes = {"goal": 0, "hole": 0}

    for episode in range(episodes):
        state = env.reset()
        state_index = env.state_to_index(state)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy strategy for action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state_index])  # Exploit

            next_state, reward, done, _ = env.step(action)
            next_state_index = env.state_to_index(next_state)

            # Q-table update
            old_value = q_table[state_index, action]
            next_max = np.max(q_table[next_state_index])
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state_index, action] = new_value

            state_index = next_state_index
            total_reward += reward

            # Render the environment if required
            if render:
                if env.render() == 'quit':
                    print("Quitting...")
                    pygame.quit()
                    return

        episode_rewards[episode] = total_reward
        # Dynamically adjust epsilon decay based on performance
        current_epsilon_decay = adjust_epsilon_decay(episode_rewards, current_epsilon_decay)
        epsilon = max(epsilon * current_epsilon_decay, 0.01)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        if track_outcomes:
            if reward > 0:  # Agent reached the goal
                outcomes["goal"] += 1
            elif reward < 0:  # Agent fell into a hole
                outcomes["hole"] += 1

        if is_training:
            with open("frozen_lake_enhanced.pkl", "wb") as f:
                pickle.dump(q_table, f)

    return (episode_rewards, outcomes) if track_outcomes else episode_rewards


if __name__ == '__main__':
    # Environment and learning parameters
    size = 8
    hole_probability = 0.12
    episodes = 10000
    test_episodes = 100
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    initial_epsilon_decay = 0.995

    # Initialize the environment and Q-table
    env = FrozenLakeEnhancedEnv(size=size, hole_probability=hole_probability, random_map=True, slippery=True)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Run training and testing episodes
    training_rewards = run_episodes(env, q_table, episodes, learning_rate, discount_factor, epsilon, initial_epsilon_decay,
                                    is_training=True, render=False)
    test_rewards = run_episodes(env, q_table, test_episodes, learning_rate, discount_factor, epsilon, initial_epsilon_decay,
                                is_training=True, render=True, track_outcomes=True)

    # Plot and save cumulative rewards
    cumulative_rewards = np.cumsum(training_rewards)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, episodes + 1), cumulative_rewards, label='Cumulative Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('frozen_lake8x8_enhanced.png')
    plt.show()

    # Clean up Pygame resources
    pygame.quit()
