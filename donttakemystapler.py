from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from ddqn import ddqn

env_dir = "/Users/andrewromans/Dev/UnityProjects/DontTakeMyStapler/DontTakeMyStapler"

unity_env = UnityEnvironment(env_dir, worker_id=0, base_port=5004)
env = UnityToGymWrapper(unity_env, uint8_visual=True)

ddqn.train_ddqn(env)

# TESTING THE ENVIRONMENT
# for episode in range(10):
#   inital_observation = env.reset()
#   done = False
#   episode_rewards = 0
#   action = env.action_space.sample()
#   while not done:
#     obs, r, done, _ = env.step(action)
#     action = env.action_space.sample()
#     print(f"{obs} | {r} | {done}")
#     episode_rewards += r

#   print("Total reward this episode: {}".format(episode_rewards))