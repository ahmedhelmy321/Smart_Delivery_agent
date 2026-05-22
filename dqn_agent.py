import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.cnn_q_network import CNNQNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_shape=(4, 16, 20),
        num_actions=5,
        lr=5e-5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
        batch_size=64,
        buffer_capacity=50000,
        target_update_freq=1000,
        device=None
    ):
        self.num_actions = num_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = CNNQNetwork(
            input_channels=state_shape[0],
            num_actions=num_actions
        ).to(self.device)

        self.target_net = CNNQNetwork(
            input_channels=state_shape[0],
            num_actions=num_actions
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr
        )

        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(buffer_capacity)

        self.learn_step_counter = 0

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def train_step(self):
        """
        One DQN update.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            1.0
        )
        self.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return loss.item()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

    def save(self, path):
        torch.save(
            self.policy_net.state_dict(),
            path
        )

    def load(self, path):
        self.policy_net.load_state_dict(
            torch.load(path)
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )