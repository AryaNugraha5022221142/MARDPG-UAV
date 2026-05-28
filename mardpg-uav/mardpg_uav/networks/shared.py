"""
Shared feature extractor with CNN for rangefinder and FC for angles/goal.
Reference: Section 6.1 and 10.1 of blueprint.
"""
import torch
import torch.nn as nn
import numpy as np

class SharedFeatureExtractor(nn.Module):
    def __init__(self, conv_filters=32, fc_att_units=16,
                 fc_prev_units=8, fc_goal_units=8):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_filters, kernel_size=2, stride=1)
        conv_out = conv_filters * 4 * 4  # 512

        self.fc_att  = nn.Linear(4, fc_att_units)   # sin/cos × 2 angles → 16
        self.fc_prev = nn.Linear(2, fc_prev_units)   # prev action → 8  (new)
        self.fc_goal = nn.Linear(3, fc_goal_units)   # goal → 8

        self.feature_dim = conv_out + fc_att_units + fc_prev_units + fc_goal_units  # 544

    def forward(self, obs):
        # obs layout: [0:4) att, [4:6) prev_act, [6:31) lidar, [31:34) goal
        att  = obs[:, 0:4]
        prev = obs[:, 4:6]
        lid  = obs[:, 6:31].view(-1, 1, 5, 5)
        goal = obs[:, 31:34]

        conv_out = torch.relu(self.conv(lid)).flatten(1)    # (B, 512)
        att_out  = torch.relu(self.fc_att(att))             # (B, 16)
        prev_out = torch.relu(self.fc_prev(prev))           # (B, 8)
        goal_out = torch.relu(self.fc_goal(goal))           # (B, 8)

        return torch.cat([conv_out, att_out, prev_out, goal_out], dim=1)  # (B, 544)
