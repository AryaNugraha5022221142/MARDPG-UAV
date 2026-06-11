"""
Shared feature extractor for the 49-D observation (obs v3).
Layout: attitude(0:4) + lidar 5x5(4:29) + goal[d, sin/cos varpi, sin/cos varpi_z](29:34)
        + neighbors K=2 x [rel_pos_body(3), rel_vel_body(3), present](34:48)
        + alive(48, omitted from the actor encoder).
"""
import torch, torch.nn as nn
import torch.nn.functional as F

class SharedFeatureExtractor(nn.Module):
    NBR_FEATS = 7  # body-frame rel pos(3) + body-frame rel vel(3) + presence(1)

    def __init__(self, obs_dim=49, n_neighbors=2):
        super(SharedFeatureExtractor, self).__init__()
        self.n_neighbors = n_neighbors
        self.fc_angle   = nn.Linear(4, 8)
        self.conv_lidar = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.lidar_out  = nn.Linear(32 * 4 * 4, 32)
        self.fc_goal    = nn.Linear(5, 8)
        self.fc_nbr     = nn.Linear(self.NBR_FEATS * n_neighbors, 16)   # 14 -> 16
        self.fc_out     = nn.Linear(8 + 32 + 8 + 16, 64)
        self.feature_dim = 64

    def forward(self, obs):
        nbr_end = 34 + self.NBR_FEATS * self.n_neighbors
        angles = obs[..., 0:4]
        lidar  = obs[..., 4:29].reshape(*obs.shape[:-1], 1, 5, 5)
        goal   = obs[..., 29:34]
        nbr    = obs[..., 34:nbr_end]
        # alive flag (index nbr_end) is intentionally omitted from the encoder.

        h_angle = F.relu(self.fc_angle(angles))
        h_lidar = F.relu(self.conv_lidar(lidar.flatten(0, -4)))
        h_lidar = F.relu(self.lidar_out(h_lidar.flatten(1)))
        h_lidar = h_lidar.view(*obs.shape[:-1], 32)
        h_goal  = F.relu(self.fc_goal(goal))
        h_nbr   = F.relu(self.fc_nbr(nbr))

        h = torch.cat([h_angle, h_lidar, h_goal, h_nbr], dim=-1)
        return F.relu(self.fc_out(h))

