#!/bin/env python3

import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from pan_radiographs_data import FullRadiographDataset
from tqdm import tqdm

from autoencoder import Autoencoder
from utils import *

if __name__ == "__main__":
    root_dir = "/datasets/pan-radiographs/"
    batch_size = 64
    epochs = 100
    bottleneck_dim = 2048
    input_size = 224
    input_channels = 1
    learning_rate = 1e-5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transforms = T.Compose([
        T.Resize((input_size,input_size)),
        T.Grayscale(),
        T.ToTensor(),
        #T.Normalize((.5,.5,.5),(.5,.5,.5)),
    ])

    # Load data
    train_set = FullRadiographDataset(root_dir, list(range(1, 21)), transforms)
    test_set  = FullRadiographDataset(root_dir, list(range(21, 31)), transforms)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(train_set, batch_size, shuffle=True)

    # Load model, optimizer and criterion
    model = Autoencoder(input_size=input_size, bottleneck_dim=bottleneck_dim, input_channels=input_channels).to(device)
    optimizer = AdamW(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    # Define metrics
    metrics = [
        "total",
        "loss",
    ]
    running_metrics = dict.fromkeys(metrics, 0)
    best_metrics = dict.fromkeys(metrics, np.inf)

    print("[!] Running on", device)
    for epoch in range(epochs):
        for x, _ in tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]"):
            x = x.to(device)

            x_hat = model(x)
            loss = criterion(x_hat, x)

            running_metrics["loss"] += loss.item()*x.size(0)
            running_metrics["total"] += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch}/{epochs}] Loss {running_metrics['loss']/running_metrics['total']:.2f}")

        if loss < best_metrics["loss"]:
            print(f"[!] New Best Loss: {best_metrics['loss']} -> {loss}. ", end='')
            best_metrics["loss"] = loss
            plt.imshow(x_hat.cpu().detach().numpy()[0][0], cmap='grey')
            plt.savefig(f"samples/epoch-{epoch}.png")
            save_checkpoint(
                filename="best_loss-2048",
                model=model,
                optimizer=optimizer,
                current_epoch=epoch
                )
