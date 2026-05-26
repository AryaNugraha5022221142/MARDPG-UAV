"""
Centralized critic network.
Input: all agents' LSTM hidden states + all agents' actions.
Reference: Section 6.2 and 5.1 of blueprint.
"""
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, n_agents: int, agent_hidden_dim: int = 128,
                 fc1_units: int = 256, lstm_hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.agent_hidden_dim = agent_hidden_dim
        
        # Input: concatenation of all agents' hidden states and actions
        # Total dim: n_agents * hidden_dim + n_agents * action_dim(2)
        input_dim = n_agents * agent_hidden_dim + n_agents * 2
        
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.lstm = nn.LSTM(fc1_units, lstm_hidden, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden, 1)
        
    def forward(self, agent_hiddens: list, actions: list,
                hidden: tuple = None) -> tuple:
        """
        Args:
            agent_hiddens: list of (batch, hidden_dim) for each agent
            actions: list of (batch, 2) for each agent
            hidden: LSTM hidden state
        Returns:
            q_value: (batch, 1)
            hidden: updated LSTM state
        """
        # Concatenate all agents' information (Section 5.1)
        joint = torch.cat(agent_hiddens + actions, dim=-1)
        
        # FC layer
        x = torch.relu(self.fc1(joint))
        
        # Add sequence dimension for LSTM processing
        x = x.unsqueeze(1)  # (batch, 1, fc1_units)
        x, hidden = self.lstm(x, hidden)
        x = x[:, -1, :]       # (batch, lstm_hidden)
        
        return self.fc_out(x), hidden
