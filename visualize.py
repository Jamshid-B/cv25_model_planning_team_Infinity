import torch
import matplotlib.pyplot as plt

from dataset import VideoDataset
from world_model import WorldModel

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cpu"
LATENT_DIM = 256
ACTION_DIM = 4

MODEL_PATH = "world_model.pth"
DATA_PATH = "data/frames"

# -------------------------
# DENORMALIZATION
# -------------------------
def denorm(x):
    """
    Convert [-1, 1] → [0, 1] for visualization
    """
    x = (x + 1.0) / 2.0
    return x.clamp(0, 1)

# -------------------------
# LOAD DATA
# -------------------------
dataset = VideoDataset(DATA_PATH)

obs, gt = dataset[0]  # take first example
obs = obs.unsqueeze(0).to(DEVICE)
gt = gt.unsqueeze(0).to(DEVICE)

# -------------------------
# LOAD MODEL
# -------------------------
model = WorldModel(
    latent_dim=LATENT_DIM,
    action_dim=ACTION_DIM
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# RUN PREDICTION
# -------------------------
with torch.no_grad():
    h = torch.zeros(1, LATENT_DIM).to(DEVICE)
    action = torch.zeros(1, ACTION_DIM).to(DEVICE)
    pred, _ = model(obs, action, h)

# -------------------------
# PREPARE IMAGES
# -------------------------
# input frame = first 3 channels of stacked input
input_img = denorm(obs[0][:3]).permute(1, 2, 0).cpu().numpy()
gt_img    = denorm(gt[0]).permute(1, 2, 0).cpu().numpy()
pred_img  = denorm(pred[0]).permute(1, 2, 0).cpu().numpy()

# -------------------------
# VISUALIZE
# -------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_img)
plt.title("Input frame (t)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gt_img)
plt.title("Ground truth (t+1)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_img)
plt.title("Predicted (t+1)")
plt.axis("off")

plt.tight_layout()
plt.show()
