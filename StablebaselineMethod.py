import gym_pendrogone   
from stable_baselines3 import PPO, a2c, SAC 
from stable_baselines3.common.env_util import make_vec_env
import gym


env = gym.make("Pendrogone-v1")
# env = make_vec_env("Pendrogone-v1", n_envs=4)

# # model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./a2c_cartpole_tensorboard/",learning_rate=1e-3)
# model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=300000)
# model.save("Pendrogone_sac")

 
# del model # remove to demonstrate saving and loading

model = SAC.load("Pendrogone_sac")
# model.learn(total_timesteps=1000000)
obs = env.reset()
env.state[2] = -1.5
env.state[3] = 2.4
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
