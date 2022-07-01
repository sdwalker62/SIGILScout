from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from ddqn import ddqn

env_dir = "/Users/andrewromans/Dev/UnityProjects/DontTakeMyStapler/DontTakeMyStapler"

unity_env = UnityEnvironment(env_dir, worker_id=0, base_port=5004)
env = UnityToGymWrapper(unity_env, uint8_visual=True)

actor = ddqn.agent(env)

def parse_actions(action):
  final_action = [0, 0, 0]
  if (action == 0 or action == 3 or action == 6):
    return final_action
  
  if (action == 1):
    return [1, 0, 0]
  elif (action == 2):
    return [2, 0, 0]
  elif (action == 4):
    return [0, 1, 0]
  elif (action == 5):
    return [0, 2, 0]
  elif (action == 7):
    return [0, 0, 1]
  elif (action == 8):
    return [0, 1, 2]

# TESTING THE ENVIRONMENT
for epoch in range(30):
  epoch_rewards = 0
  for episode_step in range(10):
    inital_observation = env.reset()
    done = False
    episode_rewards = 0
    # STOP, F, B - STOP, R, L - STOP, Rc, Lc
    action = parse_actions(actor.act(inital_observation))
    while not done:
      obs, r, done, _ = env.step(action)
      action = parse_actions(actor.act(inital_observation))
      # print(f"{obs} | {r} | {done}")
      episode_rewards += r
    print(f"Total reward for episode {episode_step}: {episode_rewards}")
    epoch_rewards += episode_rewards
  for train_step in range(10):
    actor.train()
  print(f"Total reward this epoch {epoch}: {epoch_rewards}")

actor.save_model()