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
        self.conv_lidar = nn.Conv2d(1, 32, kernel_size=2, stride=1)  # 5x5 -> 4x4x32
        self.lidar_out  = nn.Linear(32 * 4 * 4, 32)
        self.fc_goal = nn.Linear(3, 8)     # Indices 29:32
        import torch.nn.functional as F
        self.fc_out = nn.Linear(48, 64)    # 8 + 32 + 8 = 48
        self.feature_dim = 64

    def forward(self, obs):
        import torch.nn.functional as F
        # [BUG-3 FIX] Correct slicing
        angles = obs[..., 0:4]
        lidar = obs[..., 4:29].reshape(*obs.shape[:-1], 1, 5, 5)
        goal = obs[..., 29:32]

        h_angle = F.relu(self.fc_angle(angles))
        h_lidar = F.relu(self.conv_lidar(lidar.flatten(0, -4)))   # handle (B*T,1,5,5) or general
        h_lidar = F.relu(self.lidar_out(h_lidar.flatten(1)))
        h_lidar = h_lidar.view(*obs.shape[:-1], 32)  # unflatten back to matched dimensions
        h_goal = F.relu(self.fc_goal(goal))

        h_concat = torch.cat([h_angle, h_lidar, h_goal], dim=-1)
        out = F.relu(self.fc_out(h_concat))
        return out
