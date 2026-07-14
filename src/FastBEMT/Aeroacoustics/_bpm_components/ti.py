"""Component computation for BPM turbulence-ingestion noise."""

import numpy as np
import torch


@torch.inference_mode()
def compute_ti_noise(
    frequencies: torch.Tensor,
    chord: torch.Tensor,
    alpha: torch.Tensor,
    u: torch.Tensor,
    m: torch.Tensor,
    rho: torch.Tensor,
    a_inf: torch.Tensor,
    base_val_le: torch.Tensor,
    base_val_low: torch.Tensor,
    turbulence_length_scale: float,
    turbulence_intensity: float,
) -> torch.Tensor:
    """Compute turbulence-ingestion broadband noise."""
    n_sections = alpha.shape[0]
    bv_le = base_val_le[None, :, :, :, :]
    bv_low = base_val_low[None, :, :, :, :]

    f_co = 10.0 * u / (np.pi * chord)
    k1_val = 2.0 * np.pi * frequencies[:, None] / u[None, :]
    k1_bar = k1_val * chord[None, :] * 0.5

    f_co_5d = f_co[None, None, :, None, None]
    freq_2d = frequencies[:, None, None, None, None]
    bv_ti = torch.where(freq_2d < f_co_5d, bv_low, bv_le)

    beta_sq = 1.0 - m**2
    k1_beta = k1_bar / beta_sq[None, :]
    denom = 2.0 * np.pi * k1_beta + 1.0 / (1.0 + 2.4 * k1_beta)
    s_sq = 1.0 / denom
    lfc = 10.0 * s_sq * m[None, :] * (k1_bar**2) / beta_sq[None, :]
    lfc_term = torch.clamp(lfc / (1.0 + lfc), min=1e-15)
    lfc_5d = lfc_term[:, None, :, None, None]

    k1_hat = k1_val / (3.0 / (4.0 * turbulence_length_scale))
    phi_term = (k1_hat**3) / ((1.0 + k1_hat**2) ** (7.0 / 3.0))
    alpha_sq = (alpha**2)[None, None, :, None, None]

    inner_val = (
        (rho**2)
        * (a_inf**4)
        * turbulence_length_scale
        * 0.5
        * (turbulence_intensity**2)
        * phi_term.view(frequencies.shape[0], 1, n_sections, 1, 1)
        * bv_ti
    )

    spl_ti = (
        10.0 * torch.log10(torch.clamp(inner_val, min=1e-20))
        + 78.4
        + 10.0 * torch.log10(1.0 + 9.0 * alpha_sq)
        + 10.0 * torch.log10(lfc_5d)
    )

    return 10.0 ** (spl_ti / 10.0)
