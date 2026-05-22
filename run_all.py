import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from env.delivery_env import DeliveryEnv
from agents.Q_Learning_agent import QLearningAgent
from agents.double_dqn_agent import DoubleDQNAgent

EPISODES = 500


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================
# GIF SAVER
# ==========================
def save_episode_gif(frames, filename="episode.gif", fps=5):
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved: {filename}")


# ==========================
# TRAIN Q-LEARNING
# ==========================
def train_q_learning(agent, env):
    rewards = []

    for episode in range(EPISODES):
        observation, _ = env.reset()
        state = agent.preprocess_state(observation)

        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = agent.preprocess_state(next_observation)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

    return rewards


# ==========================
# TRAIN DOUBLE DQN
# ==========================
def train_double_dqn(agent, env):
    rewards = []

    for episode in range(EPISODES):
        state, _ = env.reset()

        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.train_step()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

    return rewards


# ==========================
# RECORD Q-LEARNING EPISODE
# ==========================
def record_q_learning_episode(agent, env, filename="q_learning_episode.gif"):
    frames = []

    observation, _ = env.reset()
    state = agent.preprocess_state(observation)

    done = False

    frames.append(env.render())

    while not done:
        action = agent.choose_action(state)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = agent.preprocess_state(next_observation)

        frames.append(env.render())

        state = next_state

    save_episode_gif(frames, filename)


# ==========================
# RECORD DOUBLE DQN EPISODE
# ==========================
def record_double_dqn_episode(agent, env, filename="double_dqn_episode.gif"):
    frames = []

    state, _ = env.reset()
    done = False

    frames.append(env.render())

    while not done:
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frames.append(env.render())

        state = next_state

    save_episode_gif(frames, filename)


# ==========================
# RUN Q-LEARNING
# ==========================
def run_q_learning():
    print("Running Q-Learning...")

    env = DeliveryEnv()

    agent = QLearningAgent(
        action_size=env.action_space.n
    )

    train_q_learning(agent, env)

    # greedy evaluation
    agent.epsilon = 0

    record_q_learning_episode(
        agent,
        env,
        "q_learning_episode.gif"
    )


# ==========================
# RUN DOUBLE DQN
# ==========================
def run_double_dqn():
    print("Running Double DQN...")

    env = DeliveryEnv()

    agent = DoubleDQNAgent(
        state_shape=(4, env.height, env.width),
        num_actions=env.action_space.n
    )

    train_double_dqn(agent, env)

    # greedy evaluation
    agent.epsilon = 0

    record_double_dqn_episode(
        agent,
        env,
        "double_dqn_episode.gif"
    )


# ==========================
# MAIN
# ==========================
def main():
    print("=" * 50)
    print("DELIVERY RL GIF VISUALIZATION")
    print("=" * 50)

    run_q_learning()
    run_double_dqn()

    print("Done.")


if __name__ == "__main__":
    main()