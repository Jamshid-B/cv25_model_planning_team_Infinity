print("TRAIN.PY STARTED")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from dataset import VideoDataset
from world_model import WorldModel

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cpu"
LATENT_DIM = 256
ACTION_DIM = 4
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 1

# -------------------------
# DATA
# -------------------------
dataset = VideoDataset("data/frames")
print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# -------------------------
# MODEL
# -------------------------
model = WorldModel(
    latent_dim=LATENT_DIM,
    action_dim=ACTION_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
l1_loss = nn.L1Loss()

# -------------------------
# TRAINING
# -------------------------
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")
    model.train()

    h = torch.zeros(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    total_loss = 0.0

    for i, (obs, target) in enumerate(loader):
        obs = obs.to(DEVICE)
        target = target.to(DEVICE)

        action = torch.zeros(obs.size(0), ACTION_DIM).to(DEVICE)

        pred, h = model(obs, action, h.detach())

        loss_l1 = l1_loss(pred, target)
        loss_ssim = 1 - ssim(pred, target, data_range=2.0)
        loss = loss_l1 + 0.5 * loss_ssim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.6f}")

# -------------------------
# SAVE
# -------------------------
torch.save(model.state_dict(), "world_model.pth")
print("TRAINING FINISHED SUCCESSFULLY")
