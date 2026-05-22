import os
import numpy as np
import matplotlib.pyplot as plt

from env.delivery_env import DeliveryEnv
from agents.double_dqn_agent import DoubleDQNAgent


EPISODES = 300
SAVE_DIR = "checkpoints"
SAVE_PATH = os.path.join(SAVE_DIR, "double_dqn_best.pth")

os.makedirs(SAVE_DIR, exist_ok=True)


def moving_average(data, window=20):
    if len(data) < window:
        return data

    return np.convolve(
        data,
        np.ones(window) / window,
        mode="valid"
    )


def train():
    env = DeliveryEnv()

    agent = DoubleDQNAgent(
        state_shape=(4, 16, 20),
        num_actions=env.action_space.n
    )

    rewards_history = []
    loss_history = []

    best_reward = -float("inf")

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()

        done = False
        episode_reward = 0
        episode_losses = []

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

            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward

        rewards_history.append(episode_reward)

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)

        agent.decay_epsilon()

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(SAVE_PATH)

        print(
            f"Episode {episode}/{EPISODES} | "
            f"Reward: {episode_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    plot_results(rewards_history, loss_history)


def plot_results(rewards, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.plot(
        range(19, len(rewards)),
        moving_average(rewards),
        label="Moving Avg"
    )
    plt.legend()
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    train()