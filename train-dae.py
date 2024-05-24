#!/bin/env python3

import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder import Autoencoder

def save_checkpoint(
        filename: str,
        model: torch.nn.modules.module.Module,
        optimizer: torch.optim.Optimizer,
        loss_history: list,
        current_epoch: int # In order to know how many epochs the model has been trained for
    ) -> None:

    if not filename.endswith(".pt"):
        filename += ".pt"

    print(f"Saving checkpoing to file '{filename}'...", end='')
    checkpoint = {
        "state_dict": model.state_dict,
        "optmizer": optimizer.state_dict,
        "epoch": current_epoch,
        "loss_history": loss_history
    }
    torch.save(checkpoint, filename)
    print("Saved.")


if __name__ == "__main__":
    root_dir = "/datasets/glomerulos-normal/"
    batch_size = 64
    epochs = 1000
    bottleneck_dim = 1024
    input_size = 224
    input_channels = 3
    learning_rate = 0.0001
    checkpoint_name = "dae-glomerulos-1000e"
    noise_factor = .3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()

    transforms = T.Compose([
        T.Resize((input_size,input_size)),
        #T.Grayscale(),
        T.ToTensor(),
        T.Normalize((.5,.5,.5),(.5,.5,.5)),
    ])

    # Load data
    train_set = ImageFolder(os.path.join(root_dir, 'train'), transform=transforms)
    val_set  = ImageFolder(os.path.join(root_dir, 'test'), transform=transforms)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(train_set, batch_size, shuffle=True)

    # Load model, optimizer and criterion
    model = Autoencoder(input_size=input_size, bottleneck_dim=bottleneck_dim, input_channels=input_channels).to(device)
    optimizer = Adam(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    # Define metrics
    metrics = [
        "total",
        "loss",
    ]
    running_metrics = dict.fromkeys(metrics, 0)
    best_metrics = dict.fromkeys(metrics, np.inf)
    fixed_image_for_sampling = train_set[0][0].view(1,input_channels, input_size, input_size).to(device)
    losses = []

    print("[!] Running on", device)
    for epoch in range(epochs):
        model.train()
        for it, (x, _) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]")):
            x = x.to(device)
            noise = torch.rand_like(x, device=device) 
            corrupted_x = x + noise_factor * noise
            corrupted_x = torch.clip(corrupted_x, 0., 1.)
            x_hat = model(corrupted_x)
            loss = criterion(x_hat, x)

            running_metrics["loss"] += loss.item()*x.size(0)
            running_metrics["total"] += x.size(0)
            losses.append(loss.item())
            
            writer.add_scalar("Loss/train", loss.item(), it)
            if it % 50: writer.add_image("Sample/train", model(fixed_image_for_sampling)[0], it)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch}/{epochs}] Loss {running_metrics['loss']/running_metrics['total']:.2f}")
        model.eval()
        val_metrics = dict.fromkeys(metrics, 0)
        for it, (x, _) in enumerate(tqdm(val_loader, desc="[Validation]")):
            x = x.to(device)
            
            x_hat = model(x)
            val_loss = criterion(x_hat, x)

            val_metrics["loss"] += val_loss.item()*x.size(0)
            val_metrics["total"] += x.size(0)

            writer.add_scalar("Loss/val", val_loss.item(), it)

        print(f"Validation Loss {val_metrics['loss']/val_metrics['total']:.2f}")

        if val_loss < best_metrics["loss"]:
            print(f"[!] New Best Val Loss: {best_metrics['loss']} -> {loss}. ", end='')
            best_metrics["loss"] = val_loss
            plt.imshow(x_hat.cpu().detach().numpy()[0][0], cmap='grey')
            plt.savefig(f"samples/sample-{epoch}.png")
            save_checkpoint(
                filename="best_loss-" + checkpoint_name,
                model=model,
                optimizer=optimizer,
                current_epoch=epoch,
                loss_history=losses,
                )
