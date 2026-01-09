import torch
import torch.nn as nn
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc = torch.nn.Linear(latent_dim, 256 * 8 * 8)

        self.net = torch.nn.Sequential(
            # 8x8 → 16x16
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            # 16x16 → 32x32
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            # 32x32 → 64x64
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            # 64x64 → 128x128
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, h):
        x = self.fc(h)
        x = x.view(-1, 256, 8, 8)
        return self.net(x)
