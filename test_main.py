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

# # ==========================================================================================
# Test safe environment setup
env = DeliveryEnv()

obs, info = env.reset()
env.render()


# # ==========================================================================================
# env = DeliveryEnv()

# obs, info = env.reset()

# done = False
# total_reward = 0

# while not done:
#     env.render()

#     action = bfs_next_action(
#         env,
#         env.agent_pos,
#         env.current_target
#     )

#     obs, reward, terminated, truncated, info = env.step(action)

#     total_reward += reward
#     done = terminated or truncated

# print("Episode finished")
# print("Total reward:", total_reward)



# import torch
# from models.cnn_q_network import CNNQNetwork

# model = CNNQNetwork()

# dummy = torch.randn(8, 4, 16, 20)

# output = model(dummy)

# print(output.shape)

# # ==========================================================================================


# import numpy as np
# from utils.replay_buffer import ReplayBuffer

# buffer = ReplayBuffer()

# dummy_state = np.zeros((4, 16, 20), dtype=np.float32)

# for _ in range(100):
#     buffer.push(
#         dummy_state,
#         1,
#         0.5,
#         dummy_state,
#         False
#     )

# print(len(buffer))

# batch = buffer.sample(32)

# for item in batch:
#     print(item.shape)
# # ==========================================================================================

# import torch
# from models.cnn_q_network import CNNQNetwork

# model = CNNQNetwork()

# dummy = torch.randn(8, 4, 16, 20)

# out = model(dummy)

# print(out.shape)