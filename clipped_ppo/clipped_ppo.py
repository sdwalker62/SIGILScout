import gym
import numpy as np
import tensorflow as tf
import yaml

with open("model_params.yaml", "r") as file:
    params = yaml.safe_load(file)


def test_reward():
    """
    Test the average reward gain from the current model.
    """
    state = env.reset()
    finished = False
    total_reward = 0
    print("Testing average reward...")
    test_limit = 0
    while not finished:
        state_input = ...  # TODO: Change state input
        action_probs = model_actor.predict(
            [state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1
        )
        action = np.argmax(action_probs)
        next_state, reward, finished, _ = env.step(action)
        state = next_state
        total_reward += reward
        test_limit += 1
        if test_limit > 20:
            break
    return total_reward


avg_reward = np.mean([test_reward() for _ in range(5)])


def get_advantages(params, values, masks, rewards):
    """
    Calculate the GAE (Generalized Advantage Estimation).

    Here gamma is the discount factor, the mask stops the model
    from training on a terminated state (once the stapler is found),
    and lambda is a smoothing parameter. For more information
    refer to the linked paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    returns = []
    gae_params = params["gae"]
    gae, lmbda, gamma = gae_params["gae"], gae_params["lmbda"], gae_params["gamma"]

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


env = football_env.create_environment(
    env_name="academy_empty_goal", representation="pixels", render=True
)


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = (
            K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val)
            * advantages
        )
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = (
            critic_discount * critic_loss
            + actor_loss
            - entropy_beta * K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        )
        return total_loss

    return loss


state = env.reset()

state_dims = env.observation_space.shape
print(state_dims)

n_actions = env.action_space.n
print(n_actions)

ppo_steps = 128

states = []
actions = []
values = []
masks = []
rewards = []
actions_probs = []
actions_onehot = []

model_actor = get_model_actor_image(input_dims=state_dims)
model_critic = get_model_critic_image(input_dims=state_dims)

for itr in range(ppo_steps):
    state_input = K.expand_dims(state, 0)
    action_dist = model_actor.predict([state_input], steps=1)
    q_value = model_critic.predict([state_input], steps=1)
    action = np.random.choice(n_actions, p=action_dist[0, :])
    action_onehot = np.zeros(n_actions)
    action_onehot[action] = 1

    observation, reward, done, info = env.step(action)
    mask = not done

    states.append(state)
    actions.append(action)
    actions_onehot.append(action_onehot)
    values.append(q_value)
    masks.append(mask)
    rewards.append(reward)
    actions_probs.append(action_dist)

    state = observation

    if done:
        env.reset()

env.close()
