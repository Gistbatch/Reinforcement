import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uuid

from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


class Regressor:
    def __init__(self, dims, learning_rate=0.1):
        self.W = np.random.randn(dims) / np.sqrt(dims)
        self.learning_rate = learning_rate

    def partial_fit(self, X ,Y):
        self.W += self.learning_rate * (Y - X.dot(self.W)).dot(X)
    
    def predict(self, features):
        return features.dot(self.W)

class FeatureTransformer:
    def __init__(self, env, n_components=1000, n_samples=20000):
        scaler = StandardScaler()
        sample_pos = 5 * np.random.random(n_samples) - 2.5
        sample_vel = 4 * np.random.random(n_samples) - 2
        sample_ppos = np.random.random(n_samples) - 0.5
        sample_pvel = 7 * np.random.random(n_samples) - 3.5
        samples = np.stack((sample_pos, sample_vel, sample_ppos, sample_pvel),
                           axis=-1)
        scaler.fit(samples)
        featureunion = FeatureUnion([('rbf1', RBFSampler(0.5, n_components)),
                                     ('rbf2', RBFSampler(1, n_components)),
                                     ('rbf3', RBFSampler(2, n_components)),
                                     ('rbf4', RBFSampler(4, n_components))])
        features = featureunion.fit_transform(scaler.transform(samples))
        self.dimensions = features.shape[1]
        self.scaler = scaler
        self.featurizer = featureunion

    def transform(self, state):
        return self.featurizer.transform(self.scaler.transform(state))


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        start = feature_transformer.transform([env.reset()])
        for _ in range(env.action_space.n):
            model = Regressor(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self, state):
        features = self.feature_transformer.transform(np.atleast_2d(state))
        predictions = [model.predict(features) for model in self.models]
        return np.stack(predictions).T

    def update(self, state, action, G):
        features = self.feature_transformer.transform(np.atleast_2d(state))
        self.models[action].partial_fit(features, [G])

    def action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.predict(state))


def plot_running_avg(rewards):
    N = len(rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = rewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def play_episode(model, epsilon, gamma=0.99):
    observation_t = model.env.reset()
    done = False
    steps = 0
    rewards = 0
    while not done:
        steps += 1
        action_t = model.action(observation_t, epsilon)
        observation_t1, reward, done, _ = model.env.step(action_t)
        rewards += reward
        if done and steps < 200:
            reward = -300
        if steps > 10000:
            break
        G = reward + gamma * model.predict(observation_t1).max()
        model.update(observation_t, action_t, G)
        observation_t = observation_t1

    return rewards


def cart_pole():
    env = gym.make('CartPole-v0')
    transformer = FeatureTransformer(env)
    model = Model(env, transformer)
    env = wrappers.Monitor(env, 'output/' + str(uuid.uuid4()))
    iterations = 100
    rewards = np.empty(iterations)
    for index in range(iterations):
        epsilon = 0.1 * (0.97**index)
        current_reward = play_episode(model, epsilon)
        rewards[index] = current_reward
        if (index + 1) % 10 == 0:
            print(
                f'episode: {index} reward {current_reward} epsilon: {epsilon}')
    print(f'avg reward for last 100 episodes: {rewards[-100:].mean()}')
    print(f'total steps {-rewards.sum()}')

    plt.plot(rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(rewards)


if __name__ == "__main__":
    cart_pole()