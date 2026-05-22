import random
import numpy as np
from collections import defaultdict


class QLearningAgent:

    def __init__(
        self,
        action_size,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):

        self.action_size = action_size

        self.lr = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size)
        )

    def preprocess_state(self, observation):

        agent_pos = tuple(np.argwhere(observation[2] == 1)[0])
        target_pos = tuple(np.argwhere(observation[3] == 1)[0])

        return (agent_pos, target_pos)

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):

        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (
            target_q - current_q
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )