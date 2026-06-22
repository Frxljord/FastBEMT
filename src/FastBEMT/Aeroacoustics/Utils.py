"""Spectral post-processing utilities for aeroacoustic pressure signals."""

from __future__ import annotations

import math

import torch

__all__ = [
    "a_weighting_db",
    "power_ratio_to_spl",
    "spl_spectrum_to_overall_level",
    "sum_spl_spectra",
    "third_octave_spectrum_to_overall_level",
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
        frequency_bin_width: Target frequency bin width in Hz. If provided,
            the pressure signal will be zero-padded to achieve the desired
            resolution. Default is None (use native FFT resolution).
    """
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
    
    # Enforce frequency bin width if specified
    if frequency_bin_width is not None:
        frequency_bin_width = float(frequency_bin_width)
        if not math.isfinite(frequency_bin_width) or frequency_bin_width <= 0.0:
            raise ValueError("frequency_bin_width must be finite and greater than zero.")
        # Required duration: T = 1 / frequency_bin_width
        # Required samples: N = ceil(T / sample_spacing)
        required_duration = 1.0 / frequency_bin_width
        required_sample_count = math.ceil(required_duration / sample_spacing)
        # Pad to required sample count if necessary
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
    if not math.isfinite(p_ref) or p_ref <= 0.0:
        raise ValueError("p_ref must be finite and greater than zero.")
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
        if not math.isfinite(floor_value) or floor_value <= 0.0:
            raise ValueError("floor must be finite and greater than zero.")
    return 10.0 * torch.log10(power_ratio.clamp_min(floor_value))


def sum_spl_spectra(
    spl: torch.Tensor,
    *,
    component_dim: int = 0,
) -> torch.Tensor:
    """Log-sum SPL spectra along a component dimension."""
    spl = torch.as_tensor(spl)
    if not spl.is_floating_point():
        spl = spl.to(torch.get_default_dtype())
    if spl.ndim == 0:
        raise ValueError("spl must have at least one dimension.")

    component_dim = component_dim % spl.ndim
    power_ratio = torch.pow(10.0, spl / 10.0)
    return power_ratio_to_spl(torch.sum(power_ratio, dim=component_dim))


def time_domain_to_spl_spectrum(
    pressure: torch.Tensor,
    sample_spacing: float,
    p_ref: float = 20.0e-6,
    *,
    time_dim: int = -1,
    frequency_bin_width: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pressure histories to a one-sided RMS SPL spectrum.

    Args:
        pressure: Time-domain acoustic pressure in Pa.
        sample_spacing: Time between samples in seconds.
        p_ref: Reference acoustic pressure in Pa.
        time_dim: Dimension containing time samples.
        frequency_bin_width: Target frequency bin width in Hz. Pressure signals
            will be zero-padded to achieve this resolution. Default is 20 Hz.

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


def third_octave_spectrum_to_overall_level(
    spl: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    weighted: bool = False,
    frequency_dim: int = -1,
) -> torch.Tensor:
    """Integrate third-octave band SPL into OSPL or A-weighted OASPL."""
    return spl_spectrum_to_overall_level(
        spl,
        frequencies,
        weighted=weighted,
        frequency_dim=frequency_dim,
    )
