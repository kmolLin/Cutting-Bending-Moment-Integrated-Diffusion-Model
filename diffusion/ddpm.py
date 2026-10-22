"""
Denoising Diffusion Probabilistic Model (DDPM) Forward and Reverse Formulations.

Implements:
- Linear variance schedule (beta_1 = 1e-4 to beta_T = 0.02, T = 1000)
- Forward diffusion process q(x_t | x_0)
- Reverse denoising process p_theta(x_{t-1} | x_t, x_sim, w, t)
"""

from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm
from models.unet1d import UNet1D


class DiffusionManager:
    """Manages the DDPM noise schedule, forward diffusion q_sample, and reverse p_sample."""

    def __init__(self, cfg, device: torch.device):
        self.device = device
        self.n_timesteps = getattr(cfg, "n_timesteps", 1000)
        beta_start = getattr(cfg, "beta_start", 1e-4)
        beta_end = getattr(cfg, "beta_end", 0.02)

        self.betas = torch.linspace(beta_start, beta_end, self.n_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Forward diffusion process: sample x_t ~ q(x_t | x_0).
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_noisy = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
        return x_noisy, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: int,
        t_index: int,
        sim_cond: torch.Tensor,
        wear_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse denoising step: sample x_{t-1} ~ p_theta(x_{t-1} | x_t, x_sim, w, t)."""
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]

        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        pred_noise = model(x, sim_cond, wear_cond, t_tensor)

        model_mean = self.sqrt_recip_alphas[t_index] * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(self.posterior_variance[t_index]) * noise


class DiffusionPredictor:
    """High-level wrapper for loading a trained checkpoint and synthesizing bending moments."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🔄 Loading pre-trained diffusion model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.stats = checkpoint["stats"]  # (mu_in, std_in, mu_out, std_out)

        config_dict = checkpoint.get("config", {})
        # Handle stringified or native values in config_dict
        parsed_cfg = {}
        for k, v in config_dict.items():
            if isinstance(v, str):
                try:
                    parsed_cfg[k] = eval(v)
                except Exception:
                    parsed_cfg[k] = v
            else:
                parsed_cfg[k] = v

        self.cfg = SimpleNamespace(**parsed_cfg)
        if not hasattr(self.cfg, "model_dim"):
            self.cfg.model_dim = 64
        if not hasattr(self.cfg, "out_channels"):
            self.cfg.out_channels = 4
        if not hasattr(self.cfg, "sim_channels"):
            self.cfg.sim_channels = 4

        self.model = UNet1D(self.cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.diff_manager = DiffusionManager(self.cfg, self.device)

    def preprocess_input(self, sim_data: np.ndarray, wear_val: float):
        """Normalize 4-channel mechanistic simulation input and scalar tool wear."""
        mu_in, std_in, _, _ = self.stats
        sim_norm = (sim_data - mu_in) / (std_in + 1e-6)
        max_wear = float(getattr(self.cfg, "max_wear", 0.4))
        wear_norm = wear_val / max_wear

        sim_tensor = (
            torch.tensor(sim_norm, dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 2, 1)
            .to(self.device)
        )
        wear_tensor = torch.tensor([[wear_norm]], dtype=torch.float32, device=self.device)
        return sim_tensor, wear_tensor

    @torch.no_grad()
    def generate(self, sim_data: np.ndarray, wear_val: float, seed: int = None) -> np.ndarray:
        """
        Synthesize realistic cutting bending moment signals given mechanistic simulation
        and tool wear state VB.

        Args:
            sim_data: Array of shape (L, 4) -> [Sim_Bx, Sim_By, Sim_R, Sim_Theta]
            wear_val: Tool flank wear VB (mm)
            seed: Optional random seed for reproducible sampling

        Returns:
            generated_exp: Array of shape (L, 4) -> [Gen_Bx, Gen_By, Gen_R, Gen_Theta]
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        sim_tensor, wear_tensor = self.preprocess_input(sim_data, wear_val)
        x = torch.randn_like(sim_tensor)

        n_steps = self.diff_manager.n_timesteps
        for t in tqdm(
            reversed(range(n_steps)),
            total=n_steps,
            leave=False,
            desc="DDPM Reverse Sampling",
        ):
            x = self.diff_manager.p_sample(
                self.model, x, t, t, sim_tensor, wear_tensor
            )

        result = x.squeeze(0).permute(1, 0).cpu().numpy()  # [L, 4]
        _, _, mu_out, std_out = self.stats
        result = result * (std_out + 1e-6) + mu_out
        return result
