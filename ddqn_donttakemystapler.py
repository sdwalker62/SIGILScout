"""
Double DQN & Natural DQN comparison,
The Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import tensorflow as tf 
import numpy as np 
import gym
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

class DDDQN_Accelerate(tf.keras.Model):
    def __init__(self):
      super(DDDQN_Accelerate, self).__init__()
      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(128, activation='relu')
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(3, activation=None)

    def call(self, input_data):
      x = self.d1(input_data)
      x = self.d2(x)
      v = self.v(x)
      a = self.a(x)
      Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
      return Q

    def advantage(self, state):
      x = self.d1(state)
      x = self.d2(x)
      a = self.a(x)
      return a

class DDDQN_Steer(tf.keras.Model):
    def __init__(self):
      super(DDDQN_Steer, self).__init__()
      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(128, activation='relu')
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(3, activation=None)

    def call(self, input_data):
      x = self.d1(input_data)
      x = self.d2(x)
      v = self.v(x)
      a = self.a(x)
      Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
      return Q

    def advantage(self, state):
      x = self.d1(state)
      x = self.d2(x)
      a = self.a(x)
      return a

class DDDQN_Brake(tf.keras.Model):
    def __init__(self):
      super(DDDQN_Brake, self).__init__()
      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(128, activation='relu')
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(3, activation=None)

    def call(self, input_data):
      x = self.d1(input_data)
      x = self.d2(x)
      v = self.v(x)
      a = self.a(x)
      Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
      return Q

    def advantage(self, state):
      x = self.d1(state)
      x = self.d2(x)
      a = self.a(x)
      return a

class exp_replay():
    def __init__(self, buffer_size= 1000000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *(env.observation_space.shape)), dtype=np.float32)
        self.action_mem = np.zeros((3, self.buffer_size), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *(env.observation_space.shape)), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done, i):
        idx  = self.pointer % self.buffer_size 
        self.state_mem[idx] = state
        self.action_mem[i][idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = [self.action_mem[i][batch] for i in range(3)]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones



class agent():
      def __init__(self, gamma=0.99, replace=100, lr=0.001):
          self.gamma = gamma
          self.epsilon = 1.0
          self.min_epsilon = 0.01
          self.epsilon_decay = 1e-3
          self.replace = replace
          self.trainstep = 0
          self.memory = exp_replay()
          self.batch_size = 64
          self.q_net_a = DDDQN_Accelerate()
          self.q_net_s = DDDQN_Steer()
          self.q_net_b = DDDQN_Brake()
          self.target_net_a = DDDQN_Accelerate()
          self.target_net_s = DDDQN_Steer()
          self.target_net_b = DDDQN_Brake()
          opt = tf.keras.optimizers.Adam(learning_rate=lr)
          self.q_net_a.compile(loss='mse', optimizer=opt)
          self.q_net_s.compile(loss='mse', optimizer=opt)
          self.q_net_b.compile(loss='mse', optimizer=opt)
          self.target_net_a.compile(loss='mse', optimizer=opt)
          self.target_net_s.compile(loss='mse', optimizer=opt)
          self.target_net_b.compile(loss='mse', optimizer=opt)


      def act(self, state):
          if np.random.rand() <= self.epsilon:
              return [np.random.choice([i for i in range(3)]) for _ in range(3)]

          else:
              a = self.q_net_a.advantage(np.array([state]))
              s = self.q_net_s.advantage(np.array([state]))
              b = self.q_net_b.advantage(np.array([state]))
              action = [np.argmax(a), np.argmax(s), np.argmax(b)]
              return action


      
      def update_mem(self, state, action, reward, next_state, done, i):
          self.memory.add_exp(state, action, reward, next_state, done, i)


      def update_target(self):
          self.target_net_a.set_weights(self.q_net_a.get_weights())   
          self.target_net_s.set_weights(self.q_net_s.get_weights())    
          self.target_net_b.set_weights(self.q_net_b.get_weights())      

      def update_epsilon(self):
          self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
          return self.epsilon

          
      def train(self):
          if self.memory.pointer < self.batch_size:
             return 
          
          if self.trainstep % self.replace == 0:
             self.update_target()
          states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
          target_a = self.q_net_a.predict(states)
          target_s = self.q_net_s.predict(states)
          target_b = self.q_net_b.predict(states)

          batch_index = np.arange(self.batch_size, dtype=np.int32)


          next_state_val_a = self.target_net_a.predict(next_states)
          max_action_a = np.argmax(self.q_net_a.predict(next_states), axis=1)
          q_target_a = np.copy(target_a)
          q_target_a[batch_index, actions] = rewards + self.gamma * next_state_val_a[batch_index, max_action_a]*dones
          self.q_net_a.train_on_batch(states, q_target_a)

          next_state_val_s = self.target_net_s.predict(next_states)
          max_action_s = np.argmax(self.q_net_s.predict(next_states), axis=1)
          q_target_s = np.copy(target_s)
          q_target_s[batch_index, actions] = rewards + self.gamma * next_state_val_s[batch_index, max_action_s]*dones
          self.q_net_s.train_on_batch(states, q_target_s)

          next_state_val_b = self.target_net_b.predict(next_states)
          max_action_b = np.argmax(self.q_net_b.predict(next_states), axis=1)
          q_target_b = np.copy(target_b)
          q_target_b[batch_index, actions] = rewards + self.gamma * next_state_val_b[batch_index, max_action_b]*dones
          self.q_net_b.train_on_batch(states, q_target_b)

          self.update_epsilon()
          self.trainstep += 1

      # def save_model(self):
      #     self.q_net.save("model.h5")
      #     self.target_net.save("target_model.h5")


      # def load_model(self):
      #     self.q_net = load_model("model.h5")
      #     self.target_net = load_model("model.h5")

env_dir = "/Users/andrewromans/Dev/UnityProjects/DontTakeMyStapler/DontTakeMyStapler"
unity_env = UnityEnvironment(env_dir, worker_id=0, base_port=5004)
env = UnityToGymWrapper(unity_env, uint8_visual=True)

agentoo7 = agent()
steps = 400
for s in range(steps):
  done = False
  state = env.reset()
  total_reward = 0
  while not done:
    #env.render()
    action = agentoo7.act(state)
    # action = [1, 0, 0]
    next_state, reward, done, _ = env.step(action)
    for i, a in enumerate(action):
      agentoo7.update_mem(state, a, reward, next_state, done, i)
    agentoo7.train()
    state = next_state
    total_reward += reward
    
    if done:
      print("total reward after {} episode is {} and epsilon is {}".format(s, total_reward, agentoo7.epsilon))

