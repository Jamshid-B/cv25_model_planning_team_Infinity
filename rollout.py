import torch

def rollout(model, start_frame, steps=10):
    h = torch.zeros(1, 256)
    frame = start_frame
    predictions = []

    for _ in range(steps):
        action = torch.zeros(1, 4)
        frame, h = model(frame, action, h)
        predictions.append(frame)

    return predictions
