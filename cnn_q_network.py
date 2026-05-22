import torch
import torch.nn as nn


class CNNQNetwork(nn.Module):
    def __init__(self, input_channels=4, num_actions=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.q_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(32 * 8 * 10, 128),
            nn.ReLU(),

            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        return self.q_head(x)