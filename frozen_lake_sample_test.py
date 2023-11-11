import gymnasium as gym


def first_run():
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human')
    observation = env.reset()[0]
    finished = False
    redirect = False

    while not (finished and redirect):
        action = env.action_space.sample()
        new_observation, reward, finished, redirect,_ = env.step(action)
        observation = new_observation
    env.close()


if __name__ == '__main__':
    first_run()