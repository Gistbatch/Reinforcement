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

        self.scaler = scaler
        self.featurizer = featureunion

    def transform(self, state):
        return self.featurizer.transform(self.scaler.transform(state))


class Model:
    def __init__(self, env, feature_transformer, learning_rate='constant'):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []
        start = feature_transformer.transform([env.reset()])
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(start, [0])
            self.models.append(model)

    def predict(self, state):
        features = self.feature_transformer.transform([state])
        predictions = [model.predict(features) for model in self.models]
        return np.stack(predictions).T

    def update(self, state, action, G):
        features = self.feature_transformer.transform([state])
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
        if steps > 10000:
            break
        G = reward + gamma * model.predict(observation_t1).max()
        model.update(observation_t, action_t, G)
        observation_t = observation_t1

    return rewards


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


def mountain_car():
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
        if (index+1) % 100 == 0:
            print(
                f'episode: {index} reward {current_reward} epsilon: {epsilon}')
    print(f'avg reward for last 100 episodes: {rewards[-100:].mean()}')
    print(f'total steps {-rewards.sum()}')

    plt.plot(rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(rewards)
    plot_cost_to_go(env, model)


if __name__ == "__main__":
    mountain_car()