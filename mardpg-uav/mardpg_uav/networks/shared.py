"""
Shared feature extractor — faithful to Section V.B and Fig. 5.
Input layout (30D): [theta(1), phi(1), lidar(25), d5(1), varpi(1), varpi_z(1)]
"""
import torch, torch.nn as nn

class SharedFeatureExtractor(nn.Module):
    def __init__(self, conv_filters=32, fc_angle_units=8, fc_goal_units=8):
        super().__init__()
        # Rangefinder branch: 5x5 -> Conv(N, 2x2, stride 1) -> ReLU
        self.conv = nn.Conv2d(1, conv_filters, kernel_size=2, stride=1)
        conv_out  = conv_filters * 4 * 4      # 512

        # Angle branch: [theta, phi] -> FC -> 8D
        self.fc_angle = nn.Linear(2, fc_angle_units)

        # Goal branch: [d5, varpi, varpi_z] -> FC -> 8D
        self.fc_goal  = nn.Linear(3, fc_goal_units)

        self.feature_dim = conv_out + fc_angle_units + fc_goal_units  # 528

    def forward(self, obs):
        # obs layout: [0:2) angles, [2:27) lidar, [27:30) goal
        angles = obs[:, 0:2]
        lidar  = obs[:, 2:27].view(-1, 1, 5, 5)
        goal   = obs[:, 27:30]

        conv_out  = torch.relu(self.conv(lidar)).flatten(1)   # (B, 512)
        angle_out = torch.relu(self.fc_angle(angles))          # (B, 8)
        goal_out  = torch.relu(self.fc_goal(goal))             # (B, 8)

        return torch.cat([conv_out, angle_out, goal_out], dim=1)  # (B, 528)
