from env.delivery_env import DeliveryEnv
from test_agent import bfs_next_action

# env = DeliveryEnv()

# obs, info = env.reset()

# env.render()

# done = False

# while not done:
#     action = env.action_space.sample()

#     obs, reward, terminated, truncated, info = env.step(action)

#     env.render()

#     done = terminated or truncated

# ==========================================================================================
# # Test safe environment setup
# env = DeliveryEnv()

# obs, info = env.reset()
# env.render()


# ==========================================================================================
env = DeliveryEnv()

obs, info = env.reset()

done = False
total_reward = 0

while not done:
    env.render()

    action = bfs_next_action(
        env,
        env.agent_pos,
        env.current_target
    )

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated

print("Episode finished")
print("Total reward:", total_reward)
