"""
Mechanistic Cutting Bending Moment Simulator (Physical Prior Generator).

Computes analytical cutting bending moments (Sim_Bx, Sim_By) based on
end-milling kinematics, helical flute geometry, chip thickness h(phi),
and cutting/edge force coefficients (Kt, Kte, Kn, Kne, Ka, Kae).
"""

import numpy as np


def run_mechanistic_simulation(params: dict, sampling_rate: float = 2500.0):
    """
    Simulate one spindle revolution of ideal cutting bending moments (Bx, By).

    Args:
        params: Dictionary of cutting condition and tool geometry parameters:
            - Spindle_Sj: Spindle speed (RPM)
            - Feed_Rate: Feed rate (mm/min)
            - Depth_of_: Axial depth of cut ap (mm)
            - Width_of_: Radial width of cut ae (mm)
            - Tool_Diameter: Tool diameter D (mm)
            - Helixangle: Helix angle (deg)
            - Flute_Cou: Number of flutes Nt
            - Milling_Ty: Milling type (1: Down milling, 0: Up milling)
            - Kt, Kte, Kn, Kne, Ka, Kae: Cutting and edge force coefficients
            - L1(m), L2(m): Sensor moment arm distances (m)
        sampling_rate: Data acquisition sampling rate (Hz), default 2500.0 Hz.

    Returns:
        phi_grid: Angular grid (deg) of shape (step_rev,)
        sim_Bx: Simulated X-axis bending moment (Nm) of shape (step_rev,)
        sim_By: Simulated Y-axis bending moment (Nm) of shape (step_rev,)
    """
    omega = float(params["Spindle_Sj"])
    Nt = int(params["Flute_Cou"])
    step_rev = int(round(60.0 / omega * sampling_rate))
    ft = float(params["Feed_Rate"]) / omega / Nt
    ap = float(params["Depth_of_"])
    ae = float(params["Width_of_"])
    r = float(params["Tool_Diameter"]) / 2.0
    helical = float(params["Helixangle"])
    updown = int(params["Milling_Ty"])
    L1 = float(params["L1(m)"])
    L2 = float(params["L2(m)"])

    if updown == 0:  # Up Milling
        phis = 180.0 - np.degrees(np.arccos(np.clip((r - ae) / r, -1.0, 1.0)))
        phie = 180.0
    else:  # Down Milling
        phis = 0.0
        phie = np.degrees(np.arccos(np.clip((r - ae) / r, -1.0, 1.0)))

    dphi = 360.0 / step_rev
    theta_rad = np.arange(Nt) * (2.0 * np.pi / Nt)
    sin_t, cos_t = np.sin(theta_rad), np.cos(theta_rad)

    if helical == 0:
        db = ap
    else:
        db = (r * 2.0) * np.radians(dphi) / 2.0 / np.tan(np.radians(helical))

    steps_axial = max(1, int(round(ap / db)))
    L1_depth = [L1 - (db * i / 1000.0) for i in range(steps_axial)]

    bx_coef = {k: np.zeros(step_rev) for k in ["kt", "kte", "kn", "kne", "ka", "kae"]}
    by_coef = {k: np.zeros(step_rev) for k in ["kt", "kte", "kn", "kne", "ka", "kae"]}
    phi_grid = np.linspace(0, (step_rev - 1) * dphi, step_rev)
    tooth_indices = np.round((np.arange(Nt) * (360.0 / Nt)) / dphi).astype(int)

    for cnt1 in range(step_rev):
        current_teeth = (tooth_indices + cnt1) % step_rev
        Ft_sum, Fte_sum, Fn_sum, Fne_sum, Fa_sum, Fae_sum = [
            np.zeros(Nt) for _ in range(6)
        ]

        for cnt3 in range(Nt):
            for cnt4 in range(steps_axial):
                phi_idx = (current_teeth[cnt3] - cnt4) % step_rev
                phia = phi_grid[phi_idx]

                if phis <= phia <= phie:
                    h = max(0.0, ft * np.sin(np.radians(phia)))
                    Ft_local, Fte_local = L1_depth[cnt4] * db * h, L1_depth[cnt4] * db
                    Fn_local, Fne_local = L1_depth[cnt4] * db * h, L1_depth[cnt4] * db
                    Fa_local, Fae_local = db * h, db
                else:
                    Ft_local = Fte_local = Fn_local = Fne_local = Fa_local = Fae_local = 0.0

                Ft_sum[cnt3] += Ft_local
                Fte_sum[cnt3] += Fte_local
                Fn_sum[cnt3] += Fn_local
                Fne_sum[cnt3] += Fne_local
                Fa_sum[cnt3] += Fa_local
                Fae_sum[cnt3] += Fae_local

        bx_coef["kt"][cnt1] = np.sum(Ft_sum * cos_t)
        bx_coef["kte"][cnt1] = np.sum(Fte_sum * cos_t)
        bx_coef["kn"][cnt1] = np.sum(Fn_sum * sin_t)
        bx_coef["kne"][cnt1] = np.sum(Fne_sum * sin_t)
        bx_coef["ka"][cnt1] = L2 * np.sum(Fa_sum * cos_t)
        bx_coef["kae"][cnt1] = L2 * np.sum(Fae_sum * cos_t)

        by_coef["kt"][cnt1] = np.sum(Ft_sum * sin_t)
        by_coef["kte"][cnt1] = np.sum(Fte_sum * sin_t)
        by_coef["kn"][cnt1] = -np.sum(Fn_sum * cos_t)
        by_coef["kne"][cnt1] = -np.sum(Fne_sum * cos_t)
        by_coef["ka"][cnt1] = -L2 * np.sum(Fa_sum * sin_t)
        by_coef["kae"][cnt1] = -L2 * np.sum(Fae_sum * sin_t)

    sim_Bx = (
        bx_coef["kt"] * params["Kt"]
        + bx_coef["kte"] * params["Kte"]
        + bx_coef["kn"] * params["Kn"]
        + bx_coef["kne"] * params["Kne"]
        + bx_coef["ka"] * params["Ka"]
        + bx_coef["kae"] * params["Kae"]
    )
    sim_By = (
        by_coef["kt"] * params["Kt"]
        + by_coef["kte"] * params["Kte"]
        + by_coef["kn"] * params["Kn"]
        + by_coef["kne"] * params["Kne"]
        + by_coef["ka"] * params["Ka"]
        + by_coef["kae"] * params["Kae"]
    )

    return phi_grid, sim_Bx, sim_By


def align_simulation_to_experiment(
    sim_Bx: np.ndarray, sim_By: np.ndarray, exp_Bx: np.ndarray, exp_By: np.ndarray
):
    """Align single-revolution simulated signal to experimental signal via circular RMS minimization."""
    min_len = min(len(exp_Bx), len(sim_Bx))
    exp_bx_arr = exp_Bx[:min_len]
    exp_by_arr = exp_By[:min_len]

    best_shift = 0
    min_error = float("inf")
    for shift in range(len(sim_Bx)):
        shifted_bx = np.roll(sim_Bx, shift)[:min_len]
        shifted_by = np.roll(sim_By, shift)[:min_len]
        err = np.sqrt(np.mean((exp_bx_arr - shifted_bx) ** 2)) + np.sqrt(
            np.mean((exp_by_arr - shifted_by) ** 2)
        )
        if err < min_error:
            min_error = err
            best_shift = shift

    return np.roll(sim_Bx, best_shift), np.roll(sim_By, best_shift), best_shift
