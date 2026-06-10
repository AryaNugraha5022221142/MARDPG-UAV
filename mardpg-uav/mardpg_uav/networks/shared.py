"""
Shared feature extractor for the 43-D observation (obs v2).
Layout: attitude(0:4) + lidar 5x5(4:29) + goal[d, sinφ,cosφ, sinφz,cosφz](29:34)
        + neighbors K=2 x [dx,dy,dz,present](34:42) + alive(42, omitted).
"""
import torch, torch.nn as nn
import torch.nn.functional as F

class SharedFeatureExtractor(nn.Module):
    def __init__(self, obs_dim=43, n_neighbors=2):
        super(SharedFeatureExtractor, self).__init__()
        self.n_neighbors = n_neighbors
        self.fc_angle   = nn.Linear(4, 8)                       # 0:4
        self.conv_lidar = nn.Conv2d(1, 32, kernel_size=2, stride=1)  # 5x5 -> 4x4x32
        self.lidar_out  = nn.Linear(32 * 4 * 4, 32)
        self.fc_goal    = nn.Linear(5, 8)                       # 29:34
        self.fc_nbr     = nn.Linear(4 * n_neighbors, 16)        # 34:34+4K
        self.fc_out     = nn.Linear(8 + 32 + 8 + 16, 64)        # = 64
        self.feature_dim = 64

    def forward(self, obs):
        angles = obs[..., 0:4]
        lidar  = obs[..., 4:29].reshape(*obs.shape[:-1], 1, 5, 5)
        goal   = obs[..., 29:34]
        nbr    = obs[..., 34:34 + 4 * self.n_neighbors]
        # alive flag (index 34+4K) is intentionally omitted from the actor encoder.

        h_angle = F.relu(self.fc_angle(angles))
        h_lidar = F.relu(self.conv_lidar(lidar.flatten(0, -4)))
        h_lidar = F.relu(self.lidar_out(h_lidar.flatten(1)))
        h_lidar = h_lidar.view(*obs.shape[:-1], 32)
        h_goal  = F.relu(self.fc_goal(goal))
        h_nbr   = F.relu(self.fc_nbr(nbr))

        h = torch.cat([h_angle, h_lidar, h_goal, h_nbr], dim=-1)
        return F.relu(self.fc_out(h))
