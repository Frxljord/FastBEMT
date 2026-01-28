import numpy as np

class LowFidelityParameters:
    """Global propeller parameters for BEMT analysis."""
    
    def __init__(
        self, 
        rpm: float, 
        a_inf: float, 
        rho: float, 
        mu: float, 
        n_blades: int, 
        p_ref: float, 
        revolutions: int, 
        num_obs_times_per_rev: int
    ):
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
        self.src_times = np.arange(0, self.num_obs_times) * self.dt
        self.num_src_times = self.src_times.size
        self.blade_angles = 2.0 * np.pi / n_blades * np.arange(n_blades)