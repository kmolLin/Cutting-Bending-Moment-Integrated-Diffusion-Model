"""
Training Script for the Physics-Guided Diffusion Model (1D-CNN U-Net + FiLM).

Usage:
    # Train on default full dataset (or sample dataset if full dataset is absent)
    python train.py

    # Train on sample dataset for quick verification
    python train.py --data_dir ./data/sample_dataset --toolwear_csv ./data/Tool_Wear.csv \
                    --train_cond NO1 --test_cond NO10 --epochs 5
"""

import argparse
import datetime
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CutBasedDataset
from diffusion.ddpm import DiffusionManager
from models.unet1d import UNet1D


class Config:
    # Default paths (auto-fallback to ./data/sample_dataset if ./processed_csv_individual_10sec is absent)
    data_dir = (
        "./processed_csv_individual_10sec"
        if os.path.exists("./processed_csv_individual_10sec")
        else "./data/sample_dataset"
    )
    toolwear_csv = (
        "./Tool_Wear.csv" if os.path.exists("./Tool_Wear.csv") else "./data/Tool_Wear.csv"
    )
    base_save_dir = "./checkpoints"

    # Default condition split
    train_cond = (
        ["NO1", "NO2", "NO3", "NO4", "NO5", "NO6", "NO7", "NO8"]
        if os.path.exists("./processed_csv_individual_10sec")
        else ["NO1"]
    )
    test_cond = (
        ["NO9", "NO10"]
        if os.path.exists("./processed_csv_individual_10sec")
        else ["NO10"]
    )

    cut_length = 25000
    max_wear = 0.4

    # Training hyperparameters
    chunk_size = 2048
    batch_size = 32
    lr = 2e-4
    epochs = 500
    patience = 50

    # DDPM schedule hyperparameters
    n_timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # 1D-CNN U-Net architecture parameters
    sim_channels = 4
    out_channels = 4
    model_dim = 64

    experiment_dir = ""
    best_model_path = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Physics-Guided 1D-CNN U-Net Diffusion Model"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Root folder containing condition subfolders (NO1, NO2, ...)")
    parser.add_argument("--toolwear_csv", type=str, default=None, help="Path to Tool_Wear.csv")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--train_cond", nargs="+", default=None, help="List of training conditions (e.g. NO1 NO2)")
    parser.add_argument("--test_cond", nargs="+", default=None, help="List of validation/testing conditions (e.g. NO9 NO10)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--chunk_size", type=int, default=None, help="Signal window length (default: 2048)")
    return parser.parse_args()


@torch.no_grad()
def validate(model, dataloader, diff_manager, device, criterion, n_timesteps):
    model.eval()
    total_loss = 0.0
    count = 0
    for sim, exp, wear, _ in dataloader:
        sim, exp, wear = sim.to(device), exp.to(device), wear.to(device)
        t = torch.randint(0, n_timesteps, (exp.shape[0],), device=device).long()
        noise = torch.randn_like(exp)
        x_noisy, _ = diff_manager.q_sample(exp, t, noise)
        pred_noise = model(x_noisy, sim, wear, t)
        loss = criterion(pred_noise, noise)
        total_loss += loss.item()
        count += 1
    return total_loss / count if count > 0 else float("inf")


def main():
    args = parse_args()
    cfg = Config()

    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.toolwear_csv is not None:
        cfg.toolwear_csv = args.toolwear_csv
    if args.save_dir is not None:
        cfg.base_save_dir = args.save_dir
    if args.train_cond is not None:
        cfg.train_cond = args.train_cond
    if args.test_cond is not None:
        cfg.test_cond = args.test_cond
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.experiment_dir = os.path.join(cfg.base_save_dir, f"run_{timestamp}")
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    cfg.best_model_path = os.path.join(cfg.experiment_dir, "model_best.pth")

    config_dict = {
        k: getattr(cfg, k)
        for k in dir(cfg)
        if not k.startswith("__") and not callable(getattr(cfg, k))
    }
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using Device: {device}")
    print(f"📂 Experiment Directory: {cfg.experiment_dir}")

    diff_manager = DiffusionManager(cfg, device)

    train_ds = CutBasedDataset(cfg, cfg.train_cond, mode="train")
    stats = train_ds.get_stats()
    test_ds = CutBasedDataset(cfg, cfg.test_cond, stats=stats, mode="test")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = UNet1D(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    print(f"\n🚀 Starting Training (Parameters: {sum(p.numel() for p in model.parameters()):,})")
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        for sim, exp, wear, _ in loop:
            sim, exp, wear = sim.to(device), exp.to(device), wear.to(device)
            t = torch.randint(0, cfg.n_timesteps, (exp.shape[0],), device=device).long()
            noise = torch.randn_like(exp)
            x_noisy, _ = diff_manager.q_sample(exp, t, noise)

            optimizer.zero_grad()
            pred_noise = model(x_noisy, sim, wear, t)
            loss = criterion(pred_noise, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dl)
        avg_val_loss = validate(
            model, test_dl, diff_manager, device, criterion, cfg.n_timesteps
        )
        scheduler.step(avg_val_loss)

        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stats": stats,
                    "config": config_dict,
                },
                cfg.best_model_path,
            )
            save_msg = "🔥 Best Saved!"
        else:
            early_stop_counter += 1

        print(
            f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} {save_msg}"
        )

        if early_stop_counter >= cfg.patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n✅ Training complete. Best checkpoint saved to: {cfg.best_model_path}")


if __name__ == "__main__":
    main()
