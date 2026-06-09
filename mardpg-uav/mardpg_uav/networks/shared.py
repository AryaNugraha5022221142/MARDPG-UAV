"""
Shared feature extractor — faithful to Section V.B and Fig. 5.
Input layout (30D): [theta(1), phi(1), lidar(25), d5(1), varpi(1), varpi_z(1)]
"""
import torch, torch.nn as nn

class SharedFeatureExtractor(nn.Module):
    def __init__(self, obs_dim=32):
        super(SharedFeatureExtractor, self).__init__()
        # [BUG-3 FIX] Correct input dimensions based on 32-D layout
        self.fc_angle = nn.Linear(4, 8)    # Indices 0:4
        self.fc_lidar = nn.Linear(25, 32)  # Indices 4:29
        self.fc_goal = nn.Linear(3, 8)     # Indices 29:32
        self.fc_out = nn.Linear(48, 64)    # 8 + 32 + 8 = 48
        self.feature_dim = 64

    def forward(self, obs):
        # [BUG-3 FIX] Correct slicing
        angles = obs[..., 0:4]
        lidar = obs[..., 4:29]
        goal = obs[..., 29:32]

        h_angle = torch.nn.functional.relu(self.fc_angle(angles))
        h_lidar = torch.nn.functional.relu(self.fc_lidar(lidar))
        h_goal = torch.nn.functional.relu(self.fc_goal(goal))

        h_concat = torch.cat([h_angle, h_lidar, h_goal], dim=-1)
        out = torch.nn.functional.relu(self.fc_out(h_concat))
        return out
