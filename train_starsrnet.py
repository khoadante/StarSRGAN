import os
import shutil

import torch
from torch.cuda import amp
from utils.model_validators import validate_starsrnet

from utils.quality_assessment import NIQE
from utils.dataset_loaders import load_datasets
from utils.model_trainers import train_starsrnet
from utils.model_losses import define_starsrnet_loss
from utils.model_builders import build_starsrnet_model
from utils.model_optimizers import define_starsrnet_optimizer
from utils.model_schedulers import define_starsrnet_scheduler
import config


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_datasets()
    print("Load all datasets successfully.")

    model, ema_model = build_starsrnet_model()
    print("Build all model successfully.")

    pixel_criterion = define_starsrnet_loss()
    print("Define all loss functions successfully.")

    optimizer = define_starsrnet_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_starsrnet_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume, map_location=lambda storage, loc: storage
        )
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_state_dict.keys()
        }
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        # Load ema model state dict. Extract the fitted model weights
        ema_model_state_dict = ema_model.state_dict()
        ema_state_dict = {
            k: v
            for k, v in checkpoint["ema_state_dict"].items()
            if k in ema_model_state_dict.keys()
        }
        # Overwrite the model weights to the current model (ema model)
        ema_model_state_dict.update(ema_state_dict)
        ema_model.load_state_dict(ema_model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the optimizer scheduler
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    niqe_model = NIQE(config.upscale_factor, config.niqe_model_path)

    # Transfer the IQA model to the specified device
    niqe_model = niqe_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        train_starsrnet(
            model,
            ema_model,
            train_prefetcher,
            pixel_criterion,
            optimizer,
            epoch,
            scaler,
        )
        _ = validate_starsrnet(
            model, ema_model, valid_prefetcher, epoch, niqe_model, "Valid"
        )
        niqe = validate_starsrnet(
            model, ema_model, test_prefetcher, epoch, niqe_model, "Test"
        )
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = niqe < best_niqe
        best_niqe = min(niqe, best_niqe)
        torch.save(
            {
                "epoch": epoch + 1,
                "best_niqe": best_niqe,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_best.pth.tar"),
            )
        if (epoch + 1) == config.epochs:
            shutil.copyfile(
                os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_last.pth.tar"),
            )


if __name__ == "__main__":
    main()
