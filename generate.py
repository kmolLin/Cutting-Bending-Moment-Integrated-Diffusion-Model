"""
Inference & Signal Synthesis Script for the Physics-Guided Diffusion Model.

Synthesizes realistic cutting bending moment signals (Bx, By, M) conditioned on
deterministic mechanistic cutting bending moments and tool flank wear (VB).

Usage:
    # Quick reproduction using pre-trained weights and sample dataset (NO10, Cut #1)
    python generate.py

    # Custom condition, cutting pass, or manual wear value
    python generate.py --checkpoint ./checkpoints/pretrained_diffusion.pth \
                       --condition NO10 --cut_idx 20 --output_dir ./outputs
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.dataset import build_4channel_features
from diffusion.ddpm import DiffusionPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize Physics-Guided Cutting Bending Moments via DDPM"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/pretrained_diffusion.pth",
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/sample_dataset",
        help="Directory containing condition folders (default: ./data/sample_dataset)",
    )
    parser.add_argument(
        "--toolwear_csv",
        type=str,
        default="./data/Tool_Wear.csv",
        help="Path to Tool_Wear.csv",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="NO10",
        help="Target cutting condition folder name (e.g., NO10)",
    )
    parser.add_argument(
        "--cut_idx",
        type=int,
        default=1,
        help="Cutting pass index (e.g., 1 for spike_NO10_NO-1.csv)",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Optional direct path to a single input CSV file (overrides --data_dir/--condition/--cut_idx)",
    )
    parser.add_argument(
        "--wear",
        type=float,
        default=None,
        help="Optional explicit tool flank wear VB (mm). If omitted, reads from Tool_Wear.csv",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Number of sampling points to synthesize (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible reverse diffusion sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save synthesized CSV and comparison plots",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        default=True,
        help="Apply circular RMS phase alignment to ground truth for evaluation (default: True)",
    )
    parser.add_argument(
        "--no_align",
        action="store_false",
        dest="align",
        help="Disable phase alignment",
    )
    return parser.parse_args()


def align_signals(real_x: np.ndarray, real_y: np.ndarray, gen_x: np.ndarray, gen_y: np.ndarray):
    """Align generated signals to ground-truth signals via circular RMS error minimization."""
    best_shift = 0
    min_error = float("inf")
    n = len(real_x)
    for shift in range(n):
        shifted_x = np.roll(gen_x, shift)
        shifted_y = np.roll(gen_y, shift)
        total_err = np.sqrt(np.mean((real_x - shifted_x) ** 2)) + np.sqrt(
            np.mean((real_y - shifted_y) ** 2)
        )
        if total_err < min_error:
            min_error = total_err
            best_shift = shift
    return np.roll(gen_x, best_shift), np.roll(gen_y, best_shift), best_shift



def compute_metrics(real_bx_by: np.ndarray, gen_bx_by: np.ndarray):
    """Compute Time-Domain (RMSE, MAE, PCC) and Frequency-Domain (FFT Spectral PCC) metrics."""
    rmse_bx = float(np.sqrt(np.mean((real_bx_by[:, 0] - gen_bx_by[:, 0]) ** 2)))
    rmse_by = float(np.sqrt(np.mean((real_bx_by[:, 1] - gen_bx_by[:, 1]) ** 2)))
    mae_bx = float(np.mean(np.abs(real_bx_by[:, 0] - gen_bx_by[:, 0])))
    mae_by = float(np.mean(np.abs(real_bx_by[:, 1] - gen_bx_by[:, 1])))

    pcc_bx = float(np.nan_to_num(np.corrcoef(real_bx_by[:, 0], gen_bx_by[:, 0])[0, 1]))
    pcc_by = float(np.nan_to_num(np.corrcoef(real_bx_by[:, 1], gen_bx_by[:, 1])[0, 1]))

    real_m = np.sqrt(real_bx_by[:, 0] ** 2 + real_bx_by[:, 1] ** 2)
    gen_m = np.sqrt(gen_bx_by[:, 0] ** 2 + gen_bx_by[:, 1] ** 2)
    rmse_m = float(np.sqrt(np.mean((real_m - gen_m) ** 2)))
    mae_m = float(np.mean(np.abs(real_m - gen_m)))

    fft_real_bx = np.abs(np.fft.rfft(real_bx_by[:, 0]))
    fft_gen_bx = np.abs(np.fft.rfft(gen_bx_by[:, 0]))
    fft_pcc_bx = float(np.nan_to_num(np.corrcoef(fft_real_bx, fft_gen_bx)[0, 1]))

    fft_real_by = np.abs(np.fft.rfft(real_bx_by[:, 1]))
    fft_gen_by = np.abs(np.fft.rfft(gen_bx_by[:, 1]))
    fft_pcc_by = float(np.nan_to_num(np.corrcoef(fft_real_by, fft_gen_by)[0, 1]))

    fft_real_m = np.abs(np.fft.rfft(real_m))
    fft_gen_m = np.abs(np.fft.rfft(gen_m))
    fft_pcc_m = float(np.nan_to_num(np.corrcoef(fft_real_m, fft_gen_m)[0, 1]))

    return {
        "rmse_bx": rmse_bx,
        "rmse_by": rmse_by,
        "mae_bx": mae_bx,
        "mae_by": mae_by,
        "pcc_bx": pcc_bx,
        "pcc_by": pcc_by,
        "rmse_m": rmse_m,
        "mae_m": mae_m,
        "fft_pcc_bx": fft_pcc_bx,
        "fft_pcc_by": fft_pcc_by,
        "fft_pcc_m": fft_pcc_m,
        "real_m": real_m,
        "gen_m": gen_m,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve input CSV file
    if args.input_csv is not None:
        target_csv_file = args.input_csv
    else:
        target_csv_file = os.path.join(
            args.data_dir,
            args.condition,
            f"spike_{args.condition}_NO-{args.cut_idx}.csv",
        )
        if not os.path.exists(target_csv_file) and os.path.exists("./processed_csv_individual_10sec"):
            target_csv_file = os.path.join(
                "./processed_csv_individual_10sec",
                args.condition,
                f"spike_{args.condition}_NO-{args.cut_idx}.csv",
            )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(target_csv_file):
        raise FileNotFoundError(f"Signal CSV file not found: {target_csv_file}")

    # Resolve tool flank wear (VB)
    if args.wear is not None:
        real_wear_val = float(args.wear)
    else:
        if not os.path.exists(args.toolwear_csv):
            raise FileNotFoundError(f"Tool wear CSV not found: {args.toolwear_csv}")
        df_wear = pd.read_csv(args.toolwear_csv)
        if args.condition not in df_wear.columns:
            raise ValueError(f"Condition {args.condition} not found in {args.toolwear_csv}")
        wear_series = df_wear[args.condition].dropna().values
        array_idx = max(0, args.cut_idx - 1)
        real_wear_val = float(
            wear_series[array_idx] if array_idx < len(wear_series) else wear_series[-1]
        )

    print(
        f"🎯 Target Condition: {args.condition} | Cut #{args.cut_idx} | Tool Wear VB = {real_wear_val:.4f} mm"
    )
    print(f"📄 Loading Signal Data: {target_csv_file}")

    df_signal = pd.read_csv(target_csv_file)
    raw_data = df_signal.iloc[:, :4].values.astype(np.float32)

    total_len = len(raw_data)
    chunk_size = args.chunk_size
    if total_len < chunk_size:
        raise ValueError(f"Signal length ({total_len}) is smaller than chunk_size ({chunk_size}).")

    start_idx = (total_len // 2) - (chunk_size // 2)
    end_idx = start_idx + chunk_size
    chunk_data = raw_data[start_idx:end_idx]

    sim_chunk = chunk_data[:, 0:2]  # [Sim_Bx, Sim_By]
    exp_chunk = chunk_data[:, 2:4]  # [Exp_Bx, Exp_By] (Ground Truth)
    sim_input_4d = build_4channel_features(sim_chunk)

    # Synthesize signal using Physics-Guided Diffusion Model
    predictor = DiffusionPredictor(args.checkpoint)
    generated_exp = predictor.generate(sim_input_4d, wear_val=real_wear_val, seed=args.seed)

    # Optional phase alignment (eliminates initial spindle phase offset between simulation and experiment)
    best_shift = 0
    if args.align:
        gen_bx, gen_by, best_shift = align_signals(
            exp_chunk[:, 0], exp_chunk[:, 1], generated_exp[:, 0], generated_exp[:, 1]
        )
        generated_exp[:, 0] = gen_bx
        generated_exp[:, 1] = gen_by
        print(f"🔄 Phase Alignment Applied (Optimal Shift: {best_shift} steps)")

    # Compute evaluation metrics
    metrics = compute_metrics(exp_chunk, generated_exp[:, :2])

    print("\n==========================================")
    print("📊 Synthesis Evaluation Metrics:")
    print("==========================================")
    print(f"RMSE Bx: {metrics['rmse_bx']:.4f} Nm | RMSE By: {metrics['rmse_by']:.4f} Nm")
    print(f"MAE  Bx: {metrics['mae_bx']:.4f} Nm | MAE  By: {metrics['mae_by']:.4f} Nm")
    print(f"PCC  Bx: {metrics['pcc_bx']:.4f}    | PCC  By: {metrics['pcc_by']:.4f}")
    print(f"Resultant Moment M: RMSE_M = {metrics['rmse_m']:.4f} Nm | MAE_M = {metrics['mae_m']:.4f} Nm")
    print(
        f"FFT Spectral Corr:  Bx = {metrics['fft_pcc_bx']:.4f} | By = {metrics['fft_pcc_by']:.4f} | M = {metrics['fft_pcc_m']:.4f}"
    )

    # Save synthesized signal CSV
    output_csv_path = os.path.join(
        args.output_dir, f"Synthesized_{args.condition}_Cut{args.cut_idx}.csv"
    )
    sim_m = np.sqrt(sim_chunk[:, 0] ** 2 + sim_chunk[:, 1] ** 2)
    df_save = pd.DataFrame(
        {
            "Real_Bx": exp_chunk[:, 0],
            "Real_By": exp_chunk[:, 1],
            "Gen_Bx": generated_exp[:, 0],
            "Gen_By": generated_exp[:, 1],
            "Sim_Bx": sim_chunk[:, 0],
            "Sim_By": sim_chunk[:, 1],
            "Real_M": metrics["real_m"],
            "Gen_M": metrics["gen_m"],
            "Sim_M": sim_m,
        }
    )
    df_save.to_csv(output_csv_path, index=False)
    print(f"\n💾 Saved synthesized signals to: {output_csv_path}")

    # Plot comparison figure
    output_img_path = os.path.join(
        args.output_dir, f"Comparison_{args.condition}_Cut{args.cut_idx}.png"
    )
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(exp_chunk[:, 0], "k-", alpha=0.55, label="Real Experiment (GT)", linewidth=1.8)
    plt.plot(
        generated_exp[:, 0],
        "r--",
        label=f"Physics-Guided Diffusion (VB={real_wear_val:.3f} mm)",
        linewidth=1.4,
    )
    plt.plot(sim_chunk[:, 0], "g:", alpha=0.45, label="Mechanistic Prior (Sim Bx)")
    plt.title(f"{args.condition} - Cut #{args.cut_idx} - Bending Moment Bx (PCC = {metrics['pcc_bx']:.3f})")
    plt.ylabel("Bx (Nm)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(exp_chunk[:, 1], "k-", alpha=0.55, label="Real Experiment (GT)", linewidth=1.8)
    plt.plot(generated_exp[:, 1], "b--", label="Physics-Guided Diffusion", linewidth=1.4)
    plt.plot(sim_chunk[:, 1], "g:", alpha=0.45, label="Mechanistic Prior (Sim By)")
    plt.title(f"{args.condition} - Cut #{args.cut_idx} - Bending Moment By (PCC = {metrics['pcc_by']:.3f})")
    plt.ylabel("By (Nm)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(metrics["real_m"], "k-", alpha=0.55, label="Real Experiment (GT)", linewidth=1.8)
    plt.plot(metrics["gen_m"], "m--", label="Physics-Guided Diffusion", linewidth=1.4)
    plt.plot(sim_m, "g:", alpha=0.45, label="Mechanistic Prior (Sim M)")
    plt.title(
        f"{args.condition} - Cut #{args.cut_idx} - Resultant Bending Moment M (RMSE = {metrics['rmse_m']:.3f} Nm)"
    )
    plt.xlabel("Sampling Points (2500 Hz)")
    plt.ylabel("M (Nm)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=200)
    plt.close()
    print(f"🎨 Saved comparison figure to: {output_img_path}")


if __name__ == "__main__":
    main()
