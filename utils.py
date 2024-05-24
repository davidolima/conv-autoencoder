import torch

def save_checkpoint(
        filename: str,
        model: torch.nn.modules.module.Module,
        optimizer: torch.optim.Optimizer,
        current_epoch: int # In order to know how many epochs the model has been trained for
    ) -> None:

    if not filename.endswith(".pt"):
        filename += ".pt"

    print(f"Saving checkpoing to file '{filename}'...", end='')
    checkpoint = {
        "state_dict": model.state_dict,
        "optmizer": optimizer.state_dict,
        "epoch": current_epoch
    }
    torch.save(checkpoint, filename)
    print("Saved.")

def load_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: Optional[str] = None,
) -> None:
    if device is None:
        device = get_device()
        model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    # load variables
    try:
        model.load_state_dict(checkpoint['state_dict']())
    except:
        correct_state_dict = {}
        for key, val in checkpoint['state_dict']().items():
            correct_state_dict[key[len('module.'):]] = val
        model.load_state_dict(correct_state_dict)
    ## add option to pass no optimizer (e.g. in inference)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optmizer']())

def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", end=' ')
    if device == "cuda":
        print(torch.cuda.get_device_name())
    else:
        print(device)

    return device

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Few-shot learning implementation")

    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="Input batch size. (default: 64)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="Number of epochs to train the model for. (default: 100)")
    parser.add_argument("--img_size", type=int, default=256, metavar="N",
                        help="Image size. (default: 256)")
    parser.add_argument("--channels", type=int, default=1, metavar="N",
                        help="Channels. (default: 1)")
    parser.add_argument("--bottleneck_dim", type=int, default=1024, metavar="N",
                        help="Size for the bottleneck layer. (default: 1024)")
    parser.add_argument("--num_heads", type=int, default=16, metavar="N",
                        help="Number of heads for the Vision Transfomer. (default: 16)")

    parser.add_argument("--lr", type=float, default=1e-5, metavar="F",
                        help="Learning rate. (default: 1e-5)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pan-radiographs.pt", metavar="S",
                        help="Checkpoint for the encoder network. (default: `checkpoints/pan-radiographs.pt`)")

    return parser.parse_args().__dict__
