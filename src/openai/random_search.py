import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

def get_action(state, wheights):
    return 1 if state.dot(wheights) > 0 else 0

def play_episode(env, params):
    observation = env.reset()
    done = False
    length = 0
    while not done:
        length += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if length > 10000:
            break
    return length

def avg_length(env, params, epsiodes=100):
    lengths = np.empty(100)
    for epsiode in range(epsiodes):
        lengths[epsiode] = play_episode(env, params)
    return lengths.mean()

def find_params(env):
    best_params = None
    lengths = []
    best = 0
    for tries in range(100):
        params = np.random.random(4) * 2 - 1
        avg = avg_length(env, params)
        lengths.append(avg)
        if avg > best:
            best = avg
            best_params = params
    return lengths, best_params

def param_search():
    env = gym.make('CartPole-v0')
    lengths, params = find_params(env)
    plt.plot(lengths)
    plt.show()
    env = wrappers.Monitor(env, '../../output', force=True)
    avg = avg_length(env, params)
    print(f'Average of best params is: {avg}')

if __name__ == "__main__":
    param_search()