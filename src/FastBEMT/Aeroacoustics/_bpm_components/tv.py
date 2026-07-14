"""Auxiliary correlations for BPM tip-vortex noise."""

import numpy as np
import torch
from scipy.interpolate import Akima1DInterpolator

from .._bpm_common import safe_log10, st


ASPECT_DATA = np.array([2.0, 2.67, 4.0, 6.0, 12.0, 24.0])
ARATIO_DATA = np.array([0.54, 0.62, 0.71, 0.79, 0.89, 0.95])


def calc_l_tip(chord: torch.Tensor, alpha_tip: torch.Tensor) -> torch.Tensor:
    """Compute tip-vortex length scale."""
    return chord * 0.008 * alpha_tip


def aspect_ratio_correction(aspect_ratio: torch.Tensor) -> torch.Tensor:
    """Compute the tip lift-curve correction."""
    aspect_value = float(aspect_ratio.detach().cpu())
    if aspect_value < 2.0:
        aratio = 0.5
    elif aspect_value <= 24.0:
        aratio = float(Akima1DInterpolator(ASPECT_DATA, ARATIO_DATA)(aspect_value))
    else:
        aratio = 1.0
    return torch.tensor(
        aratio,
        dtype=aspect_ratio.dtype,
        device=aspect_ratio.device,
    )


@torch.inference_mode()
def compute_tv_noise(
    frequencies: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor,
    chord: torch.Tensor,
    alpha: torch.Tensor,
    m: torch.Tensor,
    a_inf: torch.Tensor,
    base_val_te: torch.Tensor,
) -> torch.Tensor:
    """Compute tip-vortex broadband noise."""
    n_source_times, n_sections, n_blades, n_observers = base_val_te.shape

    chord_tip = chord[-1]
    m_tip = m[-1]
    dr_tip = dr[-1]
    area = torch.trapz(chord, r)
    cbar = area / (r[-1] - r[0])
    alpha_tip = aspect_ratio_correction(r[-1] / cbar) * alpha[-1]

    l_tip = calc_l_tip(chord_tip, alpha_tip)
    m_max = m_tip * (1 + 0.036 * alpha_tip)

    st_freq = st(frequencies, l_tip, (a_inf * m_max).unsqueeze(0))
    st_2p = st_freq[:, :1, None, None, None]

    bv_te_tip = base_val_te[:, -1:, :, :][None, :, :, :, :]
    dh_over_r2_tip = bv_te_tip / torch.clamp(
        m_tip**5 * dr_tip,
        min=torch.finfo(base_val_te.dtype).tiny,
    )
    m_factor = m_tip**2 * m_max**3 * l_tip**2
    log_st = safe_log10(torch.abs(st_2p))

    spl_tip = (
        10 * safe_log10(m_factor * dh_over_r2_tip)
        - 30.5 * (log_st + 0.3) ** 2
        + 126
    )

    spp_full = torch.zeros(
        (
            frequencies.shape[0],
            n_source_times,
            n_sections,
            n_blades,
            n_observers,
        ),
        device=base_val_te.device,
        dtype=base_val_te.dtype,
    )
    spp_full[:, :, -1, :, :] = 10 ** (spl_tip.squeeze(2) / 10)

    return spp_full
