import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uuid

from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


class FeatureTransformer:
    def __init__(self, env, n_components=500):
        scaler = StandardScaler()
        samples = np.array(
            [env.observation_space.sample() for _ in range(10000)])
        scaler.fit(samples)
        featureunion = FeatureUnion([('rbf1', RBFSampler(0.5, n_components)),
                                     ('rbf2', RBFSampler(1, n_components)),
                                     ('rbf3', RBFSampler(2, n_components)),
                                     ('rbf4', RBFSampler(4, n_components))])
        features = featureunion.fit_transform(scaler.transform(samples))
        self.dims = features.shape[1]
        self.scaler = scaler
        self.featurizer = featureunion

    def transform(self, state):
        return self.featurizer.transform(self.scaler.transform(state))


class BasseModel():
    def __init__(self, dims):
        self.W = np.random.random(dims) / np.sqrt(dims)

    def partial_fit(self, X, Y, eligibility, learning_rate=0.01):
        self.W += learning_rate * (Y - X.dot(self.W)) * eligibility

    def predict(self, X):
        return np.array(X).dot(self.W)


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        self.eligibilities = np.zeros(
            (env.action_space.n, feature_transformer.dims))
        for _ in range(env.action_space.n):
            model = BasseModel(feature_transformer.dims)
            self.models.append(model)

    def predict(self, state):
        features = self.feature_transformer.transform([state])
        predictions = [model.predict(features) for model in self.models]
        return np.stack(predictions).T

    def update(self, state, action, G, gamma, lambda_):
        features = self.feature_transformer.transform([state])
        self.eligibilities *= gamma * lambda_
        self.eligibilities[action] += features[0]
        self.models[action].partial_fit(features[0], [G],
                                        self.eligibilities[action])

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


def play_episode(model, epsilon, gamma=0.99, lambda_=0.7):
    state = model.env.reset()
    done = False
    steps = 0
    totalreward = 0
    while not done and steps < 10000:
        action = model.action(state, epsilon)
        current_state = state
        state, reward, done, _ = model.env.step(action)

        G = reward + gamma * model.predict(state).max()
        model.update(current_state, action, G, gamma, lambda_)

        totalreward += reward
        steps += 1
    return totalreward


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0],
                    num=num_tiles)
    y = np.linspace(env.observation_space.low[1],
                    env.observation_space.high[1],
                    num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2,
                            np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def td_lambda():
    env = gym.make('MountainCar-v0')
    transformer = FeatureTransformer(env)
    model = Model(env, transformer)
    env = wrappers.Monitor(env, 'output/' + str(uuid.uuid4()))
    iterations = 300
    rewards = np.empty(iterations)
    for index in range(iterations):
        epsilon = 0.1 * (0.97**index)
        current_reward = play_episode(model, epsilon)
        rewards[index] = current_reward
        print(f'episode: {index} reward {current_reward} epsilon: {epsilon}')
    print(f'avg reward for last 100 episodes: {rewards[-100:].mean()}')
    print(f'total steps {-rewards.sum()}')

    plt.plot(rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(rewards)
    plot_cost_to_go(env, model)


if __name__ == "__main__":
    td_lambda()