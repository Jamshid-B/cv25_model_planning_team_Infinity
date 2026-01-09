import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder

        self.files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

    def __len__(self):
        # we use idx, idx+1, idx+2 → input
        # idx+3 → target
        return len(self.files) - 4

    def __getitem__(self, idx):
        paths = [
            os.path.join(self.folder, self.files[idx + i])
            for i in range(4)
        ]

        imgs = []
        for p in paths:
            img = cv2.imread(p)
            img = cv2.resize(img, (128, 128))
            img = torch.tensor(img).permute(2, 0, 1).float()
            img = img / 255.0
            img = img * 2.0 - 1.0   # [-1, 1]
            imgs.append(img)

        # stack first 3 frames → 9 channels
        input_frames = torch.cat(imgs[:3], dim=0)
        target_frame = imgs[3]

        return input_frames, target_frame
