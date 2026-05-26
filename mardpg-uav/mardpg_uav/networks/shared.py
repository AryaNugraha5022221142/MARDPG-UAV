"""
Shared feature extractor with CNN for rangefinder and FC for angles/goal.
Reference: Section 6.1 and 10.1 of blueprint.
"""
import torch
import torch.nn as nn
import numpy as np


class SharedFeatureExtractor(nn.Module):
    def __init__(self, conv_filters: int = 32, fc_angle_units: int = 8,
                 fc_goal_units: int = 8):
        super().__init__()
        # CNN for 5x5 rangefinder input
        # Input: (batch, 1, 5, 5)
        self.conv = nn.Conv2d(1, conv_filters, kernel_size=2, stride=1)
        self.conv_flatten = nn.Flatten()
        # Output: (batch, 32 * 4 * 4) = (batch, 512)
        conv_out_size = conv_filters * 4 * 4
        
        # FC for internal angles [theta, phi]
        self.fc_angle = nn.Linear(2, fc_angle_units)
        
        # FC for goal info [d_g, pi_h, pi_v]
        self.fc_goal = nn.Linear(3, fc_goal_units)
        
        self.feature_dim = conv_out_size + fc_angle_units + fc_goal_units
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, 30) or (batch*seq_len, 30)
        Returns:
            features: (batch, feature_dim)
        """
        batch_size = obs.shape[0]
        
        # Extract components (Section 3.1)
        angles = obs[:, 0:2]           # theta, phi
        rangefinder = obs[:, 2:27].view(batch_size, 1, 5, 5)  # 25 dims -> 5x5
        goal = obs[:, 27:30]           # d_g, pi_h, pi_v
        
        # Process
        conv_out = torch.relu(self.conv(rangefinder))  # (batch, 32, 4, 4)
        conv_out = self.conv_flatten(conv_out)          # (batch, 512)
        
        angle_out = torch.relu(self.fc_angle(angles))   # (batch, 8)
        goal_out = torch.relu(self.fc_goal(goal))       # (batch, 8)
        
        # Concatenate (Section 6.1)
        return torch.cat([conv_out, angle_out, goal_out], dim=1)
