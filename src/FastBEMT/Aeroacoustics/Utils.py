import numpy as np
import torch

def perform_spectral_analysis(propeller) -> None:
    '''Compute FFT, SPL, A-weighted SPL, and OASPL using PyTorch.

    Performs spectral analysis on total pressure signal with GPU acceleration.
    Computes SPL in dB, A-weighted SPL, overall SPL, and overall A-weighted SPL.
    '''
    n: int = propeller.simulation.num_obs_times
    no: int = propeller.observer_positions.shape[0]

    # Frequency grid for spectral analysis
    dt: float = (
        propeller.simulation.observer_time_range
        / propeller.simulation.num_obs_times
    )
    f_single = torch.fft.rfftfreq(n, dt).to(propeller.device)
    propeller.freq = f_single.unsqueeze(1).expand(-1, no)

    # Compute FFT and convert to RMS amplitude
    fft_p = torch.fft.rfft(propeller.p_tot, dim=0)
    propeller.fft_amp = torch.abs(fft_p)
    propeller.fft_amp.mul_(np.sqrt(2) / n)
    propeller.fft_amp[0, :].div_(np.sqrt(2))  # DC component

    # Compute SPL: 20*log10(p_rms / p_ref)
    p_ref: float = propeller.environment.p_ref
    propeller.spl = torch.clamp(propeller.fft_amp, min=1e-15)
    propeller.spl.div_(p_ref).log10_().mul_(20.0)

    def r_a_func_torch(f: torch.Tensor) -> torch.Tensor:
        """Compute A-weighting function per ISO 61672-1."""
        f_sq = f.square()
        c1 = 12194.0**2
        c2 = 20.6**2
        c3 = 107.7**2
        c4 = 737.9**2

        num = f_sq.square().mul_(c1**2)
        den = (
            f_sq.add(c2)
            .mul_(torch.sqrt(f_sq.add(c3).mul_(f_sq.add(c4))))
            .mul_(f_sq.add(c1))
        )
        return num.div_(den)

    # Normalize A-weighting to 1000 Hz
    r_a_1000 = r_a_func_torch(torch.tensor(1000.0, device=propeller.device))

    # Compute A-weighted SPL
    a_weight = r_a_func_torch(propeller.freq).div_(r_a_1000).log10_().mul_(20.0)

    propeller.spl_a = propeller.spl.clone()
    propeller.spl_a.add_(a_weight)

    # Compute Overall Sound Pressure Level (OSPL)
    p_rms_sq = propeller.fft_amp[0, :].square()
    p_rms_sq.add_(torch.sum(propeller.fft_amp[1:, :].square(), dim=0).mul_(2.0))
    propeller.ospl = torch.sqrt(p_rms_sq).div_(p_ref).log10_().mul_(20.0)
    propeller.ospl = propeller.ospl.cpu().numpy()

    # Compute Overall A-weighted Sound Pressure Level (OASPL)
    amp_a = torch.pow(10.0, propeller.spl_a.div_(20.0)).mul_(p_ref)

    p_rms_a_sq = amp_a[0, :].square()
    p_rms_a_sq.add_(torch.sum(amp_a[1:, :].square(), dim=0).mul_(2.0))
    propeller.oaspl = torch.sqrt(p_rms_a_sq).div_(p_ref).log10_().mul_(20.0)
    propeller.oaspl = propeller.oaspl.cpu().numpy()
