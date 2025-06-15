import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """An actor-critic network for the Hex game."""
    def __init__(self, board_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate the size of the flattened layer
        self._to_linear = None
        self._get_conv_output_size(board_size)
        
        # Actor and Critic heads
        self.actor_fc = nn.Linear(self._to_linear, action_size)
        self.critic_fc = nn.Linear(self._to_linear, 1)

    def _get_conv_output_size(self, board_size):
        """Helper to calculate the flattened size after conv layers."""
        x = torch.randn(1, 1, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Get policy and value
        policy = F.softmax(self.actor_fc(x), dim=-1)
        value = self.critic_fc(x)
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

class MiniResNet(nn.Module):
    """A much smaller ResNet for 7x7 boards - only 2 blocks, 16 channels."""
    def __init__(self, board_size, action_space_size):
        super(MiniResNet, self).__init__()
        num_channels = 16  # Much smaller
        num_res_blocks = 2  # Much fewer blocks
        
        self.conv_in = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy head - simplified
        self.conv_pi = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_pi = nn.BatchNorm2d(2)
        self.fc_pi = nn.Linear(2 * board_size * board_size, action_space_size)
        
        # Value head - much smaller
        self.conv_v = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(board_size * board_size, 64)  # Much smaller: 64 instead of 256
        self.fc_v2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        pi = F.relu(self.bn_pi(self.conv_pi(x)))
        pi = pi.view(pi.size(0), -1)
        pi = F.log_softmax(self.fc_pi(pi), dim=1)  # Use log_softmax for MCTS
        
        # Value head
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        
        return pi, v

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
        pi = F.log_softmax(self.fc_pi(pi), dim=1)  # Use log_softmax for MCTS
        
        # Value head
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_v1(v))
        v = torch.tanh(self.fc_v2(v))
        
        return pi, v 