"""
Dataset and Signal Pre-processing Utilities for the Physics-Guided Diffusion Model.

Handles:
- Polar coordinate feature transformation: (Bx, By) -> (Bx, By, R, theta)
- Loading cutting pass CSVs and matching tool wear labels (Tool_Wear.csv)
- Z-score normalization and random window slicing (chunk_size = 2048)
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def cart2pol(x: np.ndarray, y: np.ndarray):
    """Convert Cartesian bending moments (Bx, By) to Polar magnitude R and phase theta."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def build_4channel_features(bx_by: np.ndarray) -> np.ndarray:
    """
    Construct 4-channel physical representation [Bx, By, R, theta] from [Bx, By].
    Args:
        bx_by: numpy array of shape (L, 2)
    Returns:
        stacked: numpy array of shape (L, 4)
    """
    r, theta = cart2pol(bx_by[:, 0], bx_by[:, 1])
    return np.column_stack([bx_by[:, 0], bx_by[:, 1], r, theta])


class CutBasedDataset(Dataset):
    """
    PyTorch Dataset for loading multi-condition milling bending moment signals
    paired with mechanistic simulations and tool flank wear (VB) measurements.
    """

    def __init__(self, cfg, cond_list, stats=None, mode: str = "train"):
        self.cfg = cfg
        self.mode = mode
        self.sim_cuts = []
        self.exp_cuts = []
        self.wear_cuts = []
        self.total_possible_chunks = 0

        if os.path.exists(cfg.toolwear_csv):
            self.df_wear = pd.read_csv(cfg.toolwear_csv)
            print(f"📄 Loaded Tool Wear Labels: {cfg.toolwear_csv} {self.df_wear.shape}")
        else:
            raise FileNotFoundError(f"Missing tool wear file: {cfg.toolwear_csv}")

        print(f"[{mode.upper()}] Scanning conditions: {cond_list} in {cfg.data_dir}...")

        for cond_name in cond_list:
            folder_path = os.path.join(cfg.data_dir, cond_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Condition folder not found: {folder_path}")
                continue

            pattern = os.path.join(folder_path, f"spike_{cond_name}_NO-*.csv")
            csv_files = glob.glob(pattern)
            if not csv_files:
                print(f"⚠️ Warning: No CSV files found in {folder_path}")
                continue

            try:
                csv_files.sort(
                    key=lambda x: int(x.split("NO-")[-1].replace(".csv", ""))
                )
            except ValueError:
                csv_files.sort()

            if cond_name in self.df_wear.columns:
                wear_column = self.df_wear[cond_name].values
            else:
                wear_column = np.zeros(len(csv_files))

            for i, file_path in enumerate(csv_files):
                try:
                    df = pd.read_csv(file_path)
                    # Columns: [Sim_Bx, Sim_By, Exp_Bx, Exp_By]
                    raw_data = df.iloc[:, :4].values.astype(np.float32)

                    curr_len = len(raw_data)
                    target_len = cfg.cut_length
                    if curr_len > target_len:
                        raw_data = raw_data[:target_len]
                    elif curr_len < target_len:
                        pad_len = target_len - curr_len
                        raw_data = np.pad(raw_data, ((0, pad_len), (0, 0)), mode="edge")

                    s_data = raw_data[:, 0:2]
                    e_data = raw_data[:, 2:4]

                    sim_stacked = build_4channel_features(s_data)
                    exp_stacked = build_4channel_features(e_data)

                    # Extract Cut Index from filename if possible (e.g. spike_NO1_NO-5.csv -> idx 4)
                    try:
                        cut_num = int(os.path.basename(file_path).split("NO-")[-1].replace(".csv", ""))
                        wear_idx = cut_num - 1
                    except ValueError:
                        wear_idx = i

                    if wear_idx < len(wear_column) and not np.isnan(wear_column[wear_idx]):
                        wear_val = wear_column[wear_idx]
                    else:
                        valid_wears = wear_column[~np.isnan(wear_column)]
                        wear_val = valid_wears[-1] if len(valid_wears) > 0 else 0.0

                    wear_norm = wear_val / cfg.max_wear
                    wear_expanded = np.full((target_len, 1), wear_norm, dtype=np.float32)

                    self.sim_cuts.append(sim_stacked[np.newaxis, ...])
                    self.exp_cuts.append(exp_stacked[np.newaxis, ...])
                    self.wear_cuts.append(wear_expanded[np.newaxis, ...])

                    self.total_possible_chunks += max(1, target_len // cfg.chunk_size)

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
                    continue

        if not self.sim_cuts:
            raise RuntimeError(f"No valid CSV data found in {cfg.data_dir} for {cond_list}")

        self.all_sim = np.concatenate(self.sim_cuts, axis=0)
        self.all_exp = np.concatenate(self.exp_cuts, axis=0)
        self.all_wear = np.concatenate(self.wear_cuts, axis=0)
        self.num_cuts = self.all_sim.shape[0]
        print(f"✅ Loaded {self.num_cuts} cutting passes ({mode}).")

        if mode == "train" and stats is None:
            mu_in = self.all_sim.reshape(-1, 4).mean(axis=0)
            std_in = self.all_sim.reshape(-1, 4).std(axis=0)
            mu_out = self.all_exp.reshape(-1, 4).mean(axis=0)
            std_out = self.all_exp.reshape(-1, 4).std(axis=0)
            self.stats = (mu_in, std_in, mu_out, std_out)
        else:
            self.stats = stats
            mu_in, std_in, mu_out, std_out = stats

        self.all_sim = (self.all_sim - mu_in) / (std_in + 1e-6)
        self.all_exp = (self.all_exp - mu_out) / (std_out + 1e-6)

    def __len__(self):
        return int(self.total_possible_chunks * 2)

    def __getitem__(self, idx):
        cut_idx = np.random.randint(0, self.num_cuts)
        max_start = self.cfg.cut_length - self.cfg.chunk_size
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_idx = start_idx + self.cfg.chunk_size

        sim_chunk = self.all_sim[cut_idx, start_idx:end_idx]
        exp_chunk = self.all_exp[cut_idx, start_idx:end_idx]
        wear_chunk = self.all_wear[cut_idx, start_idx:end_idx]
        wear_val = float(np.mean(wear_chunk))

        sim_tensor = torch.tensor(sim_chunk, dtype=torch.float32).permute(1, 0)
        exp_tensor = torch.tensor(exp_chunk, dtype=torch.float32).permute(1, 0)
        wear_tensor = torch.tensor([wear_val], dtype=torch.float32)

        mask = torch.zeros((self.cfg.out_channels, self.cfg.chunk_size), dtype=torch.bool)
        return sim_tensor, exp_tensor, wear_tensor, mask

    def get_stats(self):
        return self.stats
