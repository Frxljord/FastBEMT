import numpy as np

class JobParameters:
    """Global propeller parameters for BEMT analysis."""
    
    def __init__(self, propRadius, hubRadius, nBlades, RPM, vInf, aInf, rho, mu):
        self.propRadius = propRadius
        self.propDiameter = 2 * propRadius
        self.hubRadius = hubRadius
        self.nBlades = nBlades
        self.RPM = RPM
        self.omega = 2 * np.pi * RPM / 60
        self.vInf = vInf
        self.aInf = aInf
        self.rho = rho
        self.mu = mu