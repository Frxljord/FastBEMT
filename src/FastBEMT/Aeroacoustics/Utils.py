"""Spectral post-processing utilities for aeroacoustic pressure signals."""

from __future__ import annotations

import math

import torch

__all__ = [
    "a_weighting_db",
    "power_ratio_to_spl",
    "spl_spectrum_to_overall_level",
    "time_domain_to_spl_spectrum",
]


def _one_sided_rms_spectrum(
    pressure: torch.Tensor,
    sample_spacing: float,
    time_dim: int,
    frequency_bin_width: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frequencies and one-sided RMS pressure amplitudes.
    
    Args:
        pressure: Time-domain acoustic pressure.
        sample_spacing: Time between samples in seconds.
        time_dim: Dimension containing time samples.
        frequency_bin_width: Maximum frequency-bin width in Hz. Signals with
            wider native bins are zero-padded; finer native resolution is kept.
    """
    pressure = torch.as_tensor(pressure)
    if not pressure.is_floating_point():
        pressure = pressure.to(torch.get_default_dtype())
    time_dim = time_dim % pressure.ndim
    sample_count = int(pressure.shape[time_dim])
    sample_spacing = float(sample_spacing)

    if frequency_bin_width is not None:
        frequency_bin_width = float(frequency_bin_width)
        required_sample_count = math.ceil(
            1.0 / (frequency_bin_width * sample_spacing)
        )
        if required_sample_count > sample_count:
            pad_amount = required_sample_count - sample_count
            pad_spec = [0, 0] * pressure.ndim
            pad_spec[2 * (pressure.ndim - 1 - time_dim) + 1] = pad_amount
            pressure = torch.nn.functional.pad(pressure, pad_spec)
            sample_count = required_sample_count

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
    return 20.0 * torch.log10(
        rms_amplitude.clamp_min(1.0e-15) / p_ref
    )


def power_ratio_to_spl(
    power_ratio: torch.Tensor,
    *,
    floor: float | None = None,
) -> torch.Tensor:
    """Convert an acoustic pressure-squared ratio to SPL in dB."""
    power_ratio = torch.as_tensor(power_ratio)
    if not power_ratio.is_floating_point():
        power_ratio = power_ratio.to(torch.get_default_dtype())
    if floor is None:
        floor_value = torch.finfo(power_ratio.dtype).tiny
    else:
        floor_value = float(floor)
    return 10.0 * torch.log10(power_ratio.clamp_min(floor_value))


def time_domain_to_spl_spectrum(
    pressure: torch.Tensor,
    sample_spacing: float,
    p_ref: float = 20.0e-6,
    *,
    time_dim: int = -1,
    frequency_bin_width: float | None = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pressure histories to a one-sided RMS SPL spectrum.

    Args:
        pressure: Time-domain acoustic pressure in Pa.
        sample_spacing: Time between samples in seconds.
        p_ref: Reference acoustic pressure in Pa.
        time_dim: Dimension containing time samples.
        frequency_bin_width: Maximum frequency-bin width in Hz. Signals with
            wider native bins are zero-padded. Use ``None`` to retain the
            native FFT length. Default is 20 Hz.

    Returns:
        ``(frequencies, spl)`` where ``frequencies`` is one-dimensional and
        ``spl`` has the same dimensions as ``pressure`` except that the time
        dimension contains the non-negative FFT frequencies.
    """
    frequencies, rms_amplitude = _one_sided_rms_spectrum(
        pressure,
        sample_spacing,
        time_dim,
        frequency_bin_width=frequency_bin_width,
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
    frequency_dim = frequency_dim % spl.ndim
    frequencies = torch.as_tensor(
        frequencies,
        dtype=spl.dtype,
        device=spl.device,
    )
    level_spectrum = spl
    if weighted:
        weighting_shape = [1] * spl.ndim
        weighting_shape[frequency_dim] = frequencies.numel()
        level_spectrum = spl + a_weighting_db(frequencies).reshape(
            weighting_shape
        )

    power_ratio = torch.pow(10.0, level_spectrum / 10.0)
    return power_ratio_to_spl(
        torch.sum(power_ratio, dim=frequency_dim),
    )
