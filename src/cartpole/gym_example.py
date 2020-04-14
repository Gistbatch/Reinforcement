import gym

env = gym.make('CartPole-v0')
env.reset()
box = env.observation_space
space = env.action_space

done = False
count = 0
while not done:
    count +=1
    observation, reward, done, info = env.step(space.sample())

print(f'Reached {count} steps')