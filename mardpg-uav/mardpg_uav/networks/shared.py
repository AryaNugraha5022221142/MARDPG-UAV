"""
Shared feature extractor with CNN for rangefinder and FC for angles/goal.
Reference: Section 6.1 and 10.1 of blueprint.
"""
import torch
import torch.nn as nn
import numpy as np

class SharedFeatureExtractor(nn.Module):
    def __init__(self, conv_filters=32, fc_vel_units=16,
                 fc_prev_units=8, fc_goal_units=8):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_filters, kernel_size=2, stride=1)
        conv_out = conv_filters * 4 * 4   # 512 — UNCHANGED

        self.fc_vel  = nn.Linear(3, fc_vel_units)    # velocity → 16 (was fc_att: 4→16)
        self.fc_prev = nn.Linear(3, fc_prev_units)   # prev 3D cmd → 8 (was 2→8)
        self.fc_goal = nn.Linear(3, fc_goal_units)   # goal displacement → 8 (unchanged)

        self.feature_dim = conv_out + fc_vel_units + fc_prev_units + fc_goal_units
        # = 512 + 16 + 8 + 8 = 544  ← UNCHANGED

    def forward(self, obs):
        # NEW obs layout: [0:3) vel, [3:6) prev_act, [6:31) lidar, [31:34) goal
        vel  = obs[:, 0:3]
        prev = obs[:, 3:6]
        lid  = obs[:, 6:31].view(-1, 1, 5, 5)
        goal = obs[:, 31:34]

        conv_out = torch.relu(self.conv(lid)).flatten(1)   # (B, 512)
        vel_out  = torch.relu(self.fc_vel(vel))            # (B, 16)
        prev_out = torch.relu(self.fc_prev(prev))          # (B, 8)
        goal_out = torch.relu(self.fc_goal(goal))          # (B, 8)

        return torch.cat([conv_out, vel_out, prev_out, goal_out], dim=1)  # (B, 544)
