import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """A simple CNN for Actor-Critic."""
    def __init__(self, board_size, action_space_size):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        flattened_size = 64 * board_size * board_size
        
        self.actor_fc = nn.Linear(flattened_size, action_space_size)
        self.critic_fc = nn.Linear(flattened_size, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        policy = F.softmax(self.actor_fc(x), dim=-1)
        value = torch.tanh(self.critic_fc(x)) # Value is between -1 and 1
        
        return policy, value

class ResidualBlock(nn.Module):
    """A residual block for the ResNet."""
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    """A ResNet architecture for Actor-Critic, inspired by AlphaZero."""
    def __init__(self, board_size, action_space_size, num_res_blocks=4, num_channels=64):
        super(ResNet, self).__init__()
        self.conv_in = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy head
        self.conv_pi = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_pi = nn.BatchNorm2d(2)
        self.fc_pi = nn.Linear(2 * board_size * board_size, action_space_size)
        
        # Value head
        self.conv_v = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(board_size * board_size, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        pi = F.relu(self.bn_pi(self.conv_pi(x)))
        pi = pi.view(pi.size(0), -1)
        pi = F.softmax(self.fc_pi(pi), dim=1)
        
        # Value head
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        
        return pi, v 