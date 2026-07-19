import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedFeatureExtractor(nn.Module):
    def __init__(self, obs_dim=35):
        super(SharedFeatureExtractor, self).__init__()
        self.fc_angle = nn.Linear(4, 8)
        self.conv_lidar = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.lidar_out = nn.Linear(32 * 4 * 4, 32)
        self.fc_goal = nn.Linear(5, 8)
        self.feature_dim = 64
        # Default to 49 features (including 'alive' flag)
        self.fc_out = nn.Linear(8 + 32 + 8 + 1, 64)

    def forward(self, obs):
        angles = obs[..., 0:4]
        lidar = obs[..., 4:29].reshape(*obs.shape[:-1], 1, 5, 5)
        goal = obs[..., 29:34]
        
        h_angle = F.relu(self.fc_angle(angles))
        h_lidar = F.relu(self.conv_lidar(lidar.flatten(0, -4)))
        h_lidar = F.relu(self.lidar_out(h_lidar.flatten(1)))
        h_lidar = h_lidar.view(*obs.shape[:-1], 32)
        
        h_goal = F.relu(self.fc_goal(goal))
        
        if self.fc_out.in_features == 49:
            alive = obs[..., 34:35]
            h = torch.cat([h_angle, h_lidar, h_goal, alive], dim=-1)
        else:
            h = torch.cat([h_angle, h_lidar, h_goal], dim=-1)
            
        return F.relu(self.fc_out(h))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        fc_out_weight_key = prefix + 'fc_out.weight'
        if fc_out_weight_key in state_dict:
            in_features = state_dict[fc_out_weight_key].shape[1]
            if in_features != self.fc_out.in_features:
                # Dynamically adapt to older checkpoints (e.g., 48 features instead of 49)
                self.fc_out = nn.Linear(in_features, self.fc_out.out_features)
                # Ensure the new layer is on the same device as the rest of the module
                self.fc_out.to(self.fc_angle.weight.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
