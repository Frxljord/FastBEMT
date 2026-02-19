import numpy as np
import torch


class LowFidelityParameters:
    '''Aeroacoustic simulation parameters for propeller analysis.
    
    Stores operational and environmental parameters including RPM, fluid properties,
    and time discretization for BEMT and acoustic calculations.
    '''

    def __init__(
        self,
        rpm: float,
        a_inf: float,
        rho: float,
        mu: float,
        n_blades: int,
        p_ref: float,
        revolutions: int,
        num_obs_times_per_rev: int,
        device: str,
    ) -> None:
        '''Initialize simulation parameters.
        
        Args:
            rpm: Rotational speed (revolutions per minute).
            a_inf: Speed of sound (m/s).
            rho: Fluid density (kg/m³).
            mu: Dynamic viscosity (Pa·s).
            n_blades: Number of propeller blades.
            p_ref: Reference acoustic pressure (Pa), typically 2e-5.
            revolutions: Number of propeller revolutions to simulate.
            num_obs_times_per_rev: Time steps per revolution.
            device: PyTorch device ('cuda' or 'cpu').
        '''
        self.device = torch.device(device)
        self.rpm = rpm
        self.omega = 2.0 * np.pi * rpm / 60.0
        self.a_inf = a_inf
        self.rho = rho
        self.mu = mu
        self.p_ref = p_ref
        self.revolutions = revolutions
        self.num_obs_times_per_rev = num_obs_times_per_rev
        self.duration = self.revolutions * (2.0 * np.pi / self.omega)
        self.blade_passing_period = self.duration / self.revolutions / n_blades
        self.observer_time_range = self.revolutions * self.blade_passing_period
        self.num_obs_times = self.num_obs_times_per_rev * self.revolutions
        self.dt = self.duration / self.num_obs_times
        self.src_times = (
            torch.arange(0, self.num_obs_times, dtype=torch.float32, device=self.device)
            * self.dt
        )
        self.src_times_one_rotation = (
            torch.arange(
                0, self.num_obs_times_per_rev, dtype=torch.float32, device=self.device
            )
            * self.dt
        )
        self.num_src_times = self.src_times.shape[0]
        self.blade_angles = (
            2.0
            * np.pi
            / n_blades
            * torch.arange(n_blades, dtype=torch.float32, device=self.device)
        )
