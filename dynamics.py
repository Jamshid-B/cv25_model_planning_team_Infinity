import torch
import torch.nn as nn

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=256, action_dim=4):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, latent_dim)

    def forward(self, z, action, h):
        x = torch.cat([z, action], dim=1)
        h_next = self.gru(x, h)
        return h_next
