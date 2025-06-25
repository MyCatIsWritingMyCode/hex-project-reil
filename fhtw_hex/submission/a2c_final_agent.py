import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class ActorCritic(nn.Module):
    """An actor-critic network for the Hex game. Self-contained for submission."""
    def __init__(self, board_size, action_size):
        super(ActorCritic, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        conv_output_size = self._get_conv_output_size()
        self.actor_fc = nn.Linear(conv_output_size, action_size)
        self.critic_fc = nn.Linear(conv_output_size, 1)

    def _get_conv_output_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, self.board_size, self.board_size)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        policy = F.softmax(self.actor_fc(x), dim=-1)
        value = self.critic_fc(x)
        return policy, value

class FinalA2CAgent:
    def __init__(self, board_size=7):
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.device = torch.device("cpu")
        model_filename = "a2c_agent.pth"
        model_path = os.path.join(os.path.dirname(__file__), model_filename)
        self.model = ActorCritic(self.board_size, self.action_size).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            self.model = None

    def select_move(self, board, action_set, current_player):
        if self.model is None or not action_set:
            import random
            return random.choice(list(action_set))
        board_tensor = torch.FloatTensor(np.array(board)).unsqueeze(0).unsqueeze(0).to(self.device)
        if current_player == -1:
            board_tensor *= -1
        with torch.no_grad():
            policy, _ = self.model(board_tensor)
        policy = policy.squeeze().cpu().numpy()
        valid_action_indices = [a[0] * self.board_size + a[1] for a in action_set]
        policy_mask = np.zeros_like(policy)
        policy_mask[valid_action_indices] = 1
        masked_policy = policy * policy_mask
        if masked_policy.sum() < 1e-9:
            best_action_flat = np.random.choice(valid_action_indices)
        else:
            best_action_flat = np.argmax(masked_policy)
        action_coords = (best_action_flat // self.board_size, best_action_flat % self.board_size)
        return action_coords 