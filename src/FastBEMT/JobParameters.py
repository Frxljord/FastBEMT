import numpy as np

class AerodynamicParameters:
    """Global propeller parameters for BEMT analysis."""
    
    def __init__(
        self, 
        prop_radius: float, 
        hub_radius: float, 
        n_blades: int, 
        rpm: float, 
        a_inf: float, 
        rho: float, 
        mu: float, 
    ):
        self.prop_radius = prop_radius
        self.prop_diameter = 2.0 * prop_radius
        self.hub_radius = hub_radius
        self.n_blades = n_blades
        self.rpm = rpm
        self.omega = 2.0 * np.pi * rpm / 60.0
        self.a_inf = a_inf
        self.rho = rho
        self.mu = mu
        self.blade_angles = (2.0 * np.pi / self.n_blades) * np.arange(self.n_blades)


class AcousticParameters(AerodynamicParameters):
    """Extends aerodynamic parameters with temporal settings for F1A analysis."""
    
    def __init__(
        self, 
        aero_params: AerodynamicParameters, 
        p_ref: float, 
        revolutions: int, 
        num_obs_times_per_rev: int
    ):
        self.__dict__.update(aero_params.__dict__)
        
        self.p_ref = p_ref
        self.revolutions = revolutions
        self.num_obs_times_per_rev = num_obs_times_per_rev
        self.duration = self.revolutions * (2.0 * np.pi / self.omega)
        self.blade_passing_period = self.duration / self.revolutions / self.n_blades
        self.observer_time_range = self.revolutions * self.blade_passing_period
        self.num_obs_times = self.num_obs_times_per_rev * self.revolutions
        self.dt = self.duration / self.num_obs_times 
        self.src_times = np.arange(0, self.num_obs_times) * self.dt
        self.num_src_times = self.src_times.size