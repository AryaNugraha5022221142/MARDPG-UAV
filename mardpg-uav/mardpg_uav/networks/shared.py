import torch, torch.nn as nn
import torch.nn.functional as F

class SharedFeatureExtractor(nn.Module):

    def __init__(self, obs_dim=35):
        super(SharedFeatureExtractor, self).__init__()
        self.fc_angle = nn.Linear(4, 8)
        self.conv_lidar = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.lidar_out = nn.Linear(32 * 4 * 4, 32)
        self.fc_goal = nn.Linear(5, 8)
        self.fc_out = nn.Linear(8 + 32 + 8, 64)
        self.feature_dim = 64

    def forward(self, obs):
        angles = obs[..., 0:4]
        lidar = obs[..., 4:29].reshape(*obs.shape[:-1], 1, 5, 5)
        goal = obs[..., 29:34]
        
        h_angle = F.relu(self.fc_angle(angles))
        h_lidar = F.relu(self.conv_lidar(lidar.flatten(0, -4)))
        h_lidar = F.relu(self.lidar_out(h_lidar.flatten(1)))
        h_lidar = h_lidar.view(*obs.shape[:-1], 32)
        
        h_goal = F.relu(self.fc_goal(goal))
        
        h = torch.cat([h_angle, h_lidar, h_goal], dim=-1)
        return F.relu(self.fc_out(h))
