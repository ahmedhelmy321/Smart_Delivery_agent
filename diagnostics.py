import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def moving_average(data, window=20):
    data = np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curve(rewards, title, window=20):
    plt.figure(figsize=(10, 6))

    plt.plot(rewards, alpha=0.4, label='Raw Reward')

    if len(rewards) >= window:
        smooth = moving_average(rewards, window)
        plt.plot(
            np.arange(window - 1, len(rewards)),
            smooth,
            linewidth=2,
            label=f"Moving Average ({window})"
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stability(all_rewards, title):
    rewards = np.array(all_rewards)

    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    episodes = np.arange(len(mean))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean, label="Mean Reward")

    plt.fill_between(
        episodes,
        mean - std,
        mean + std,
        alpha=0.3,
        label="±1 Std Dev"
    )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_q_heatmap(agent, env, title):
    heatmap = np.full((env.height, env.width), np.nan)

    target = env.current_target

    if target is None:
        print("No current target found.")
        return

    for r in range(env.height):
        for c in range(env.width):

            if env.base_map[r, c] == env.COMPOUND:
                continue

            if (r, c) in env.obstacles:
                continue

            state = ((r, c), target)

            q_values = agent.q_table[state]

            heatmap[r, c] = np.max(q_values)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".1f",
        cmap="viridis"
    )

    plt.title(title)
    plt.show()


def plot_policy_overlay(agent, env, title):
    frame = env.render()

    plt.figure(figsize=(8, 8))
    plt.imshow(frame)

    arrows = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→"
    }

    target = env.current_target

    if target is None:
        print("No target found.")
        return

    cell_size = 40

    for r in range(env.height):
        for c in range(env.width):

            if env.base_map[r, c] == env.COMPOUND:
                continue

            if (r, c) in env.obstacles:
                continue

            state = ((r, c), target)

            action = np.argmax(agent.q_table[state])

            plt.text(
                c * cell_size + cell_size // 2,
                r * cell_size + cell_size // 2,
                arrows[action],
                ha='center',
                va='center',
                fontsize=14,
                color='black'
            )

    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_dqn_policy_overlay(model, env, title):
    frame = env.render()

    plt.figure(figsize=(8, 8))
    plt.imshow(frame)

    arrows = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→"
    }

    cell_size = 40

    for r in range(env.height):
        for c in range(env.width):

            if env.base_map[r, c] == env.COMPOUND:
                continue

            if (r, c) in env.obstacles:
                continue

            state = env.encode_state((r, c))
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            plt.text(
                c * cell_size + cell_size // 2,
                r * cell_size + cell_size // 2,
                arrows[action],
                ha='center',
                va='center',
                fontsize=14,
                color='black'
            )

    plt.title(title)
    plt.axis('off')
    plt.show()