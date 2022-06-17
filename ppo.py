import gym
import time
import yaml
import scipy.signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train_ppo(env):
    # Open hyperparameter yaml
    with open("ppo_hyp.yaml", 'r') as file:
        hyper = yaml.load(file, Loader=yaml.FullLoader)
    
    def discounted_cumulative_sums(x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


    class Buffer:
        # Buffer for storing trajectories
        def __init__(self, observation_dimensions, size, gamma=hyper["GAMMA"], lam=hyper["LAM"]):
            # Buffer initialization
            self.observation_buffer = np.zeros(
                (size, observation_dimensions), dtype=np.float32
            )
            self.action_buffer = np.zeros(size, dtype=np.int32)
            self.advantage_buffer = np.zeros(size, dtype=np.float32)
            self.reward_buffer = np.zeros(size, dtype=np.float32)
            self.return_buffer = np.zeros(size, dtype=np.float32)
            self.value_buffer = np.zeros(size, dtype=np.float32)
            self.logprobability_buffer = np.zeros(size, dtype=np.float32)
            self.gamma, self.lam = gamma, lam
            self.pointer, self.trajectory_start_index = 0, 0

        def store(self, observation, action, reward, value, logprobability):
            # Append one step of agent-environment interaction
            self.observation_buffer[self.pointer] = observation
            self.action_buffer[self.pointer] = action
            self.reward_buffer[self.pointer] = reward
            self.value_buffer[self.pointer] = value
            self.logprobability_buffer[self.pointer] = logprobability
            self.pointer += 1

        def finish_trajectory(self, last_value=0):
            # Finish the trajectory by computing advantage estimates and rewards-to-go
            path_slice = slice(self.trajectory_start_index, self.pointer)
            rewards = np.append(self.reward_buffer[path_slice], last_value)
            values = np.append(self.value_buffer[path_slice], last_value)

            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

            self.advantage_buffer[path_slice] = discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )
            self.return_buffer[path_slice] = discounted_cumulative_sums(
                rewards, self.gamma
            )[:-1]

            self.trajectory_start_index = self.pointer

        def get(self):
            # Get all data of the buffer and normalize the advantages
            self.pointer, self.trajectory_start_index = 0, 0
            advantage_mean, advantage_std = (
                np.mean(self.advantage_buffer),
                np.std(self.advantage_buffer),
            )
            self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
            return (
                self.observation_buffer,
                self.action_buffer,
                self.advantage_buffer,
                self.return_buffer,
                self.logprobability_buffer,
            )


    def mlp(x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)


    def logprobabilities(logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
        )
        return logprobability


    # Sample action from actor
    @tf.function
    def sample_action(observation):
        logits = actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                logprobabilities(actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + hyper["CLIP_RATIO"]) * advantage_buffer,
                (1 - hyper["CLIP_RATIO"]) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - logprobabilities(actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, critic.trainable_variables)
        value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
    
    # Initialize the environment and get the dimensionality of the
    # observation space and the number of possible actions
    env = gym.make("CartPole-v1")
    observation_dimensions = env.observation_space.shape[0]
    num_actions = env.action_space.n


    # Initialize the buffer
    buffer = Buffer(observation_dimensions, hyper["STEPS_PER_EPOCH"])

    # Initialize the actor and the critic as keras models
    observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
    logits = mlp(observation_input, list(hyper["HIDDEN_SIZES"]) + [num_actions], tf.tanh, None)
    actor = keras.Model(inputs=observation_input, outputs=logits)
    value = tf.squeeze(
        mlp(observation_input, list(hyper["HIDDEN_SIZES"]) + [1], tf.tanh, None), axis=1
    )
    critic = keras.Model(inputs=observation_input, outputs=value)

    # Initialize the policy and the value function optimizers
    policy_optimizer = keras.optimizers.Adam(learning_rate=hyper["POLICY_LEARNING_RATE"])
    value_optimizer = keras.optimizers.Adam(learning_rate=hyper["VALUE_FUNCTION_LEARNING_RATE"])

    # Initialize the observation, episode return and episode length
    observation, episode_return, episode_length = env.reset(), 0, 0
    
    
    # Iterate over the number of epochs
    for epoch in range(hyper["EPOCHS"]):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        # Iterate over the steps of each epoch
        for t in range(hyper["STEPS_PER_EPOCH"]):
            if hyper["RENDER"]:
                env.render()

            # Get the logits, action, and take one step in the environment
            observation = observation.reshape(1, -1)
            logits, action = sample_action(observation)
            observation_new, reward, done, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = critic(observation)
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == hyper["STEPS_PER_EPOCH"] - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(hyper["TRAIN_POLICY_ITERATIONS"]):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * hyper["TARGET_KL"]:
                # Early Stopping
                break

        # Update the value function
        for _ in range(hyper["TRAIN_VALUE_ITERATIONS"]):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )