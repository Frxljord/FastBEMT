import numpy as np

class CompactAcousticSourceArray:
    """
    Computes vectorized F1A acoustic source terms for propeller sections and blades.
    Calculates kinematics and forces in the fixed frame based on BEMT input.
    """
    def __init__(
        self, 
        rho: float, 
        a_inf: float, 
        r: np.ndarray, 
        dr: np.ndarray, 
        area: np.ndarray, 
        chord: np.ndarray, 
        twist: np.ndarray, 
        com_shift_forward: np.ndarray, 
        com_shift_up: np.ndarray, 
        source_times: np.ndarray, 
        omega: float, 
        d_t: np.ndarray, 
        d_q: np.ndarray, 
        blade_angles: np.ndarray
    ):
        """Initialize scalars, vectorized geometry, and kinematics for all sources."""
        self.rho = rho
        self.a_inf = a_inf
        self.omega = omega

        # Geometry and loading inputs (Ns,)
        self.r = np.asarray(r)
        self.dr = np.asarray(dr)
        self.area = np.asarray(area)
        self.chord = np.asarray(chord)
        self.twist_rad = np.deg2rad(np.asarray(twist))
        self.com_shift_forward = np.asarray(com_shift_forward)
        self.com_shift_up = np.asarray(com_shift_up)
        self.d_t = np.asarray(d_t)
        self.d_q = np.asarray(d_q)

        # Time and Phase: (Nt,) and (Nb,)
        self.tau = np.atleast_1d(source_times)
        self.blade_angles = np.atleast_1d(blade_angles)

        self.nt, self.ns, self.nb = self.tau.size, self.r.size, self.blade_angles.size
        
        # Pre-compute rotation and kinematics
        self.skew_matrix = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        self._initialize_geometry_and_kinematics()

    def _initialize_geometry_and_kinematics(self):
        """Vectorized calculation of fixed-frame positions, velocities, and forces (assumed constant in the moving frame)."""
        # Source position in the moving (blade) frame (Ns, 3)
        pos_moving = np.stack([
            self.chord * self.com_shift_forward * np.sin(self.twist_rad),
            self.r,
            self.chord * self.com_shift_up * np.sin(self.twist_rad)
        ], axis=-1)

        # Rotation Matrices (Nt, Nb, 3, 3)
        angles = self.omega * self.tau[:, None] + self.blade_angles[None, :]
        c, s = np.cos(angles), np.sin(angles)
        
        rot = np.zeros((self.nt, self.nb, 3, 3))
        rot[..., 0, 0] = 1.0
        rot[..., 1, 1], rot[..., 1, 2] = c, -s
        rot[..., 2, 1], rot[..., 2, 2] = s, c

        # Transform to Fixed Frame (Nt, Ns, Nb, 3) using Einstein Summation
        self.pos_fixed = np.einsum('tbik,sk->tsbi', rot, pos_moving)
        
        # Velocity, Acceleration, Jerk via Skew Matrix multiplication
        self.vel_fixed = self.omega * np.einsum('ij,tsbj->tsbi', self.skew_matrix, self.pos_fixed)
        self.acc_fixed = self.omega * np.einsum('ij,tsbj->tsbi', self.skew_matrix, self.vel_fixed)
        self.jerk_fixed = self.omega * np.einsum('ij,tsbj->tsbi', self.skew_matrix, self.acc_fixed)

        # Forces in Fixed Frame (Nt, Ns, Nb, 3)
        force_moving = np.stack((self.d_t, np.zeros_like(self.d_t), -self.d_q), axis=-1)
        self.force_fixed = np.einsum('tbik,sk->tsbi', rot, force_moving)

    def get_observer_times(self, observers: np.ndarray) -> np.ndarray:
        """Calculate the time the sound reaches the observer (retarded time)."""
        obs = np.atleast_2d(observers)[None, None, None, :, :]  # (1, 1, 1, No, 3)
        dist = np.linalg.norm(obs - self.pos_fixed[..., None, :], axis=-1)
        return self.tau[:, None, None, None] + dist / self.a_inf

    def calculate_f1a_pressure(self, observers: np.ndarray) -> np.ndarray:
        """Compute the Thickness (monopole) and Loading (dipole) pressure components."""
        obs = np.atleast_2d(observers)[None, None, None, :, :]
        ri = obs - self.pos_fixed[..., None, :]
        r = np.linalg.norm(ri, axis=-1)
        r_hat = ri / r[..., None]

        # Mach and Mach-radial terms
        v_vec = self.vel_fixed[..., None, :]
        a_vec = self.acc_fixed[..., None, :]
        m_vec = v_vec / self.a_inf
        m_mag = np.linalg.norm(v_vec, axis=-1) / self.a_inf
        m_r = np.sum(m_vec * r_hat, axis=-1)
        
        # Helper scaling functions (Rf and derivatives)
        def get_rf(m1, m2): 
            return r**(-m1) * (1.0 - m_r)**(-m2)

        m_dot = np.sum(v_vec * a_vec, axis=-1) / (self.a_inf * np.linalg.norm(v_vec, axis=-1))
        r_hat_dot = -(self.a_inf / r)[..., None] * (m_vec - m_r[..., None] * r_hat)
        m_r_dot = (np.sum(a_vec * r_hat, axis=-1) / self.a_inf + 
                   (self.a_inf / r) * (m_r**2 - m_mag**2))

        def get_rp(m1, m2):
            return (m2 * get_rf(m1, m2+1) * m_r_dot + 
                    self.a_inf * m2 * (m_r - m_mag**2) * get_rf(m1+1, m2+1) + 
                    self.a_inf * (m1 - m2) * m_r * get_rf(m1+1, m2))

        m_r_ddot = ((np.sum(self.jerk_fixed[..., None, :] * r_hat, axis=-1) + 
                     np.sum(a_vec * r_hat_dot, axis=-1)) / self.a_inf + 
                    self.a_inf * (2 * (m_r * m_r_dot - m_mag * m_dot) / r + 
                    (m_r**2 - m_mag**2) * np.sum(r_hat * v_vec, axis=-1) / r**2))

        # Acoustic Integrands
        c1a = (get_rf(0, 2) * (self.a_inf * (get_rp(2, 2) * (m_r - m_mag**2) + 
               get_rf(2, 2) * (m_r_dot - 2*m_mag*m_dot)) + 
               m_r_ddot * get_rf(1, 2) + m_r_dot * get_rp(1, 2)) + 
               get_rf(0, 1) * get_rp(0, 1) * get_rp(1, 1))

        e1a = (get_rf(0, 1)[..., None] * (get_rp(1, 1)[..., None] * r_hat + get_rf(1, 1)[..., None] * r_hat_dot) + 
               self.a_inf * get_rf(2, 1)[..., None] * r_hat)

        # Pressure summation
        p_m = (self.rho / (4 * np.pi)) * self.area[None, :, None, None] * c1a * self.dr[None, :, None, None]
        p_d = (1.0 / (4 * np.pi * self.a_inf)) * np.sum(self.force_fixed[..., None, :] * e1a, axis=-1) * self.dr[None, :, None, None]

        return np.stack((p_m, p_d), axis=-1)