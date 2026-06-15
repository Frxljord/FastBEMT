"""Spectral post-processing utilities for aeroacoustic pressure signals."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..Propeller import Propeller

__all__ = [
    "a_weighting_db",
    "perform_spectral_analysis",
    "spl_spectrum_to_overall_level",
    "time_domain_to_spl_spectrum",
]


def _one_sided_rms_spectrum(
    pressure: torch.Tensor,
    sample_spacing: float,
    time_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frequencies and one-sided RMS pressure amplitudes."""
    pressure = torch.as_tensor(pressure)
    if not pressure.is_floating_point():
        pressure = pressure.to(torch.get_default_dtype())
    if pressure.ndim == 0:
        raise ValueError("pressure must have at least one dimension.")

    time_dim = time_dim % pressure.ndim
    sample_count = int(pressure.shape[time_dim])
    if sample_count <= 0:
        raise ValueError("The time dimension must contain at least one sample.")
    sample_spacing = float(sample_spacing)
    if not math.isfinite(sample_spacing) or sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be finite and greater than zero.")

    frequencies = torch.fft.rfftfreq(
        sample_count,
        d=sample_spacing,
        dtype=pressure.dtype,
        device=pressure.device,
    )
    rms_amplitude = torch.abs(
        torch.fft.rfft(pressure, dim=time_dim)
    ) / sample_count

    one_sided_scale = torch.ones(
        frequencies.shape,
        dtype=pressure.dtype,
        device=pressure.device,
    )
    if sample_count % 2 == 0:
        one_sided_scale[1:-1] = math.sqrt(2.0)
    else:
        one_sided_scale[1:] = math.sqrt(2.0)
    scale_shape = [1] * pressure.ndim
    scale_shape[time_dim] = frequencies.numel()
    rms_amplitude = rms_amplitude * one_sided_scale.reshape(scale_shape)
    return frequencies, rms_amplitude


def _rms_amplitude_to_spl(
    rms_amplitude: torch.Tensor,
    p_ref: float,
) -> torch.Tensor:
    """Convert RMS pressure amplitudes to SPL."""
    p_ref = float(p_ref)
    if not math.isfinite(p_ref) or p_ref <= 0.0:
        raise ValueError("p_ref must be finite and greater than zero.")
    return 20.0 * torch.log10(
        rms_amplitude.clamp_min(1.0e-15) / p_ref
    )


def time_domain_to_spl_spectrum(
    pressure: torch.Tensor,
    sample_spacing: float,
    p_ref: float = 20.0e-6,
    *,
    time_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pressure histories to a one-sided RMS SPL spectrum.

    Args:
        pressure: Time-domain acoustic pressure in Pa.
        sample_spacing: Time between samples in seconds.
        p_ref: Reference acoustic pressure in Pa.
        time_dim: Dimension containing time samples.

    Returns:
        ``(frequencies, spl)`` where ``frequencies`` is one-dimensional and
        ``spl`` has the same dimensions as ``pressure`` except that the time
        dimension contains the non-negative FFT frequencies.
    """
    frequencies, rms_amplitude = _one_sided_rms_spectrum(
        pressure,
        sample_spacing,
        time_dim,
    )
    return frequencies, _rms_amplitude_to_spl(rms_amplitude, p_ref)


def a_weighting_db(frequencies: torch.Tensor) -> torch.Tensor:
    """Return the IEC A-weighting correction for frequencies in Hz."""
    frequencies = torch.as_tensor(frequencies)
    if not frequencies.is_floating_point():
        frequencies = frequencies.to(torch.get_default_dtype())

    frequency_squared = frequencies.square()
    frequency_1000_squared = frequencies.new_tensor(1000.0).square()

    def relative_response(frequency_squared_value: torch.Tensor) -> torch.Tensor:
        return (
            12194.0**2 * frequency_squared_value.square()
            / (
                (frequency_squared_value + 20.6**2)
                * torch.sqrt(
                    (frequency_squared_value + 107.7**2)
                    * (frequency_squared_value + 737.9**2)
                )
                * (frequency_squared_value + 12194.0**2)
            )
        )

    response = relative_response(frequency_squared)
    response_1000 = relative_response(frequency_1000_squared)
    return 20.0 * torch.log10(response / response_1000)


def spl_spectrum_to_overall_level(
    spl: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    weighted: bool = False,
    frequency_dim: int = -1,
) -> torch.Tensor:
    """Integrate an SPL spectrum into OSPL or A-weighted OASPL.

    Args:
        spl: Sound pressure level spectrum in dB.
        frequencies: One-dimensional frequency vector in Hz.
        weighted: Apply A-weighting when True.
        frequency_dim: Dimension of ``spl`` corresponding to ``frequencies``.

    Returns:
        Overall level in dB with the frequency dimension removed.
    """
    spl = torch.as_tensor(spl)
    if not spl.is_floating_point():
        spl = spl.to(torch.get_default_dtype())
    if spl.ndim == 0:
        raise ValueError("spl must have at least one dimension.")

    frequency_dim = frequency_dim % spl.ndim
    frequencies = torch.as_tensor(
        frequencies,
        dtype=spl.dtype,
        device=spl.device,
    )
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be one-dimensional.")
    if frequencies.numel() != spl.shape[frequency_dim]:
        raise ValueError(
            "frequencies must match the selected frequency dimension."
        )

    level_spectrum = spl
    if weighted:
        weighting_shape = [1] * spl.ndim
        weighting_shape[frequency_dim] = frequencies.numel()
        level_spectrum = spl + a_weighting_db(frequencies).reshape(
            weighting_shape
        )

    power_ratio = torch.pow(10.0, level_spectrum / 10.0)
    return 10.0 * torch.log10(torch.sum(power_ratio, dim=frequency_dim))


def perform_spectral_analysis(propeller: Propeller) -> None:
    """Populate the established Propeller spectral result attributes."""
    f1a = getattr(propeller, "f1a", None)
    if f1a is not None and f1a.sample_spacing is not None:
        sample_spacing = f1a.sample_spacing
    else:
        sample_spacing = (
            propeller.simulation.observer_time_range
            / propeller.simulation.num_obs_times
        )
    frequencies, rms_amplitude = _one_sided_rms_spectrum(
        propeller.p_tot,
        sample_spacing,
        1,
    )
    spl = _rms_amplitude_to_spl(
        rms_amplitude,
        propeller.environment.p_ref,
    )
    observer_count = int(propeller.p_tot.shape[0])

    propeller.freq = frequencies.unsqueeze(0).expand(observer_count, -1)
    propeller.spl = spl
    propeller.spl_a = spl + a_weighting_db(frequencies)[None, :]
    propeller.ospl = (
        spl_spectrum_to_overall_level(
            spl,
            frequencies,
            frequency_dim=1,
        )
        .cpu()
        .numpy()
    )
    propeller.oaspl = (
        spl_spectrum_to_overall_level(
            spl,
            frequencies,
            weighted=True,
            frequency_dim=1,
        )
        .cpu()
        .numpy()
    )

    # Preserve the existing RMS-amplitude result for downstream callers.
    propeller.fft_amp = rms_amplitude
