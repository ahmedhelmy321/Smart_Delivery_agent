# File: experiments/train_q_learning.py

import matplotlib.pyplot as plt

from env.delivery_env import DeliveryEnv
from agents.Q_Learning_agent import QLearningAgent


EPISODES = 500


def moving_average(data, window_size=20):

    averages = []

    for i in range(len(data)):

        start = max(0, i - window_size + 1)

        avg = sum(data[start:i + 1]) / (i - start + 1)

        averages.append(avg)

    return averages


def run_q_learning():

    env = DeliveryEnv()

    agent = QLearningAgent(
        action_size=env.action_space.n
    )

    rewards_history = []

    for episode in range(EPISODES):

        observation, info = env.reset()

        state = agent.preprocess_state(observation)

        total_reward = 0

        done = False

        while not done:

            action = agent.choose_action(state)

            next_observation, reward, terminated, truncated, info = env.step(action)

            next_state = agent.preprocess_state(next_observation)

            done = terminated or truncated

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

        rewards_history.append(total_reward)

        print(
            f"Episode {episode + 1}/{EPISODES} | "
            f"Reward: {total_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    print("\nTraining Finished")

    # -----------------------------
    # Moving Average
    # -----------------------------

    moving_avg_rewards = moving_average(
        rewards_history,
        window_size=20
    )

    # -----------------------------
    # Plot Rewards
    # -----------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        rewards_history,
        label="Raw Rewards"
    )

    plt.plot(
        moving_avg_rewards,
        linewidth=3,
        label="Moving Average (20)"
    )

    plt.title("Q-Learning Training Rewards")

    plt.xlabel("Episode")

    plt.ylabel("Total Reward")

    plt.legend()

    plt.grid(True)

    plt.show()

    return agent