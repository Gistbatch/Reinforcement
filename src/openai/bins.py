import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import uuid


def plot_running_avg(rewards):
    N = len(rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = rewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


class Model:
    def __init__(self, env, feature_transformer, learning_rate=1e-2):
        self.env = env
        self.feature_transformer = feature_transformer
        self.learning_rate = learning_rate
        num_states = 10**env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1,
                                   high=1,
                                   size=(num_states, num_actions))

    def predict(self, state):
        features = self.feature_transformer.transform(state)
        return self.Q[features]

    def update(self, state, action, G):
        features = self.feature_transformer.transform(state)
        self.Q[features,action] += self.learning_rate * (G - self.Q[features,action])

    def action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.predict(state))


class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5 , 3.5, 9)

    def transform(self, state):
        cart_pos, cart_vel, pole_angle, pole_vel = state
        result = 0
        result += np.digitize([cart_pos], self.cart_position_bins)[0] * 1000
        result += np.digitize([cart_vel], self.cart_velocity_bins)[0] * 100
        result += np.digitize([pole_angle], self.pole_angle_bins)[0] * 10
        result += np.digitize([pole_vel], self.pole_velocity_bins)[0] * 1
        return result


def play_episode(model, epsilon, gamma=0.9):
    observation_t = model.env.reset()
    done = False
    steps = 0
    rewards = 0
    while not done:
        steps += 1
        action_t = model.action(observation_t, epsilon)
        observation_t1, reward, done, info = model.env.step(action_t)
        rewards += reward
        if done and steps < 200:
            reward = -300
        if steps > 10000:
            break
        G = reward + gamma * np.max(model.predict(observation_t1))
        model.update(observation_t, action_t, G)
        observation_t = observation_t1

    return rewards


def q_learning():
    env = gym.make('CartPole-v0')
    transformer = FeatureTransformer()
    model = Model(env, transformer)
    env = wrappers.Monitor(env, 'output/' + str(uuid.uuid4()))
    iterations = 10000
    rewards = np.empty(iterations)
    for index in range(iterations):
        epsilon = 1.0 / np.sqrt(index+1)
        current_reward = play_episode(model, epsilon)
        rewards[index] = current_reward
        if index % 100 == 0:
            print(f'episode: {index} reward {current_reward} epsilon: {epsilon}')
    print(f'avg reward for last 100 episodes: {rewards[-100:].mean()}')
    print(f'total steps {rewards.sum()}')

    plt.plot(rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(rewards)

if __name__ == "__main__":
    q_learning()