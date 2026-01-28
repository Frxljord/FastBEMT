import torch
import numpy as np
from typing import Union

class TorchCompactAcousticSourceArray:
    """Compact acoustic source array using FW-H equation in Mach-number formulation.
    
    This class computes acoustic pressure from a rotating blade using a torch-optimized
    compact source formulation. It handles thickness and loading noise through F1A pressure
    contributions on moving surfaces.
    """

    def __init__(
        self, 
        rho: float,
        a_inf: float, 
        r: Union[np.ndarray, torch.Tensor],
        dr: Union[np.ndarray, torch.Tensor],
        area: Union[np.ndarray, torch.Tensor],
        chord: Union[np.ndarray, torch.Tensor],
        twist: Union[np.ndarray, torch.Tensor], 
        com_shift_forward: Union[np.ndarray, torch.Tensor],
        com_shift_up: Union[np.ndarray, torch.Tensor], 
        source_times: Union[np.ndarray, torch.Tensor],
        omega: float, 
        d_t: Union[np.ndarray, torch.Tensor],
        d_q: Union[np.ndarray, torch.Tensor],
        blade_angles: Union[np.ndarray, torch.Tensor],
        device: str
    ) -> None:
        """Initialize the compact acoustic source array.
        
        Args:
            rho: Ambient fluid density (kg/m^3)
            a_inf: Ambient sound speed (m/s)
            r: Radial positions of blade sections (m)
            dr: Radial spacing of blade sections (m)
            area: Surface area of blade sections (m^2)
            chord: Chord length of blade sections (m)
            twist: Twist angle of blade sections (degrees)
            com_shift_forward: Forward center-of-mass shift (fraction of chord)
            com_shift_up: Upward center-of-mass shift (fraction of chord)
            source_times: Time array for source evaluation (s)
            omega: Blade angular velocity (rad/s)
            d_t: Thrust force distribution (N)
            d_q: Drag force distribution (N)
            blade_angles: Initial blade angles (radians)
            device: Torch device for computation ('cpu' or 'cuda')
        """
        torch.backends.opt_einsum.strategy = 'branch-all'
        self.device = torch.device(device)
        self.dtype = torch.float32

        # Ambient flow properties
        self.rho = torch.tensor(rho, dtype=self.dtype, device=self.device)
        self.a_inf = torch.tensor(a_inf, dtype=self.dtype, device=self.device)
        self.omega = torch.tensor(omega, dtype=self.dtype, device=self.device)

        # Angular velocity vector (rotates around x-axis)
        self.omega_vec = torch.zeros((1, 1, 1, 3), dtype=self.dtype, device=self.device)
        self.omega_vec[..., 0] = self.omega

        # Blade geometry arrays
        self.r = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        self.dr = torch.as_tensor(dr, dtype=self.dtype, device=self.device)
        self.area = torch.as_tensor(area, dtype=self.dtype, device=self.device)
        self.chord = torch.as_tensor(chord, dtype=self.dtype, device=self.device)
        self.twist_rad = torch.deg2rad(torch.as_tensor(twist, dtype=self.dtype, device=self.device))
        
        # Pre-compute thickness source strength: (rho / 4π) * area * dr
        # Shape: (1, Ns, 1, 1) for broadcasting in pressure summation
        thickness_strength = (self.rho / (4.0 * np.pi)) * self.area * self.dr
        self.thickness_strength = thickness_strength[None, :, None, None]
        
        # Pre-compute dipole scaling: (dr / 4π)
        # Multiplied by 1/a_inf during pressure calculation for efficiency
        dipole_strength = self.dr / (4.0 * np.pi)
        self.dipole_strength = dipole_strength[None, :, None, None]

        # Center-of-mass shift parameters
        self.com_shift_forward = torch.as_tensor(com_shift_forward, dtype=self.dtype, device=self.device)
        self.com_shift_up = torch.as_tensor(com_shift_up, dtype=self.dtype, device=self.device)

        # Force distributions
        self.d_t = torch.tensor(d_t, dtype=self.dtype, device=self.device)
        self.d_q = torch.tensor(d_q, dtype=self.dtype, device=self.device)
        self.tau = torch.as_tensor(source_times, dtype=self.dtype, device=self.device)
        self.blade_angles = torch.as_tensor(blade_angles, dtype=self.dtype, device=self.device)

        # Problem dimensions
        self.nt, self.ns, self.nb = self.tau.shape[0], self.r.shape[0], self.blade_angles.shape[0]
        
        self._initialize_geometry_and_kinematics()

    @torch.no_grad()
    def _initialize_geometry_and_kinematics(self) -> None:
        """Compute kinematic properties of rotating blade sections.
        
        Calculates position, velocity, acceleration, jerk, and force derivatives in the
        fixed reference frame by rotating sections from the moving frame. Also computes
        temporal derivatives of forces using 4th-order finite difference schemes.
        """
        # Blade section positions in moving (rotating) frame
        pos_moving = torch.stack([
            self.chord * self.com_shift_up * torch.sin(self.twist_rad),
            self.r,
            self.chord * self.com_shift_forward * torch.sin(self.twist_rad)
        ], dim=-1)

        # Compute rotation angles at each time step and blade
        angles = self.omega * self.tau[:, None] + self.blade_angles[None, :]
        c, s = torch.cos(angles), torch.sin(angles)
        
        # Build rotation matrices (rotation around x-axis)
        # Format: (time, blade, 3x3)
        rot = torch.zeros((self.nt, self.nb, 3, 3), dtype=self.dtype, device=self.device)
        rot[..., 0, 0] = 1.0
        rot[..., 1, 1], rot[..., 1, 2] = c, -s
        rot[..., 2, 1], rot[..., 2, 2] = s, c

        # Transform positions to fixed frame using rotation matrices
        # Shape: (time, section, blade, xyz)
        self.pos_fixed = torch.einsum('tbik,sk->tsbi', rot, pos_moving).contiguous()

        # Compute kinematic derivatives by successive cross products with angular velocity
        # v = ω × r, a = ω × v, j = ω × a
        self.vel_fixed = torch.linalg.cross(self.omega_vec, self.pos_fixed).contiguous()
        self.acc_fixed = torch.linalg.cross(self.omega_vec, self.vel_fixed).contiguous()
        self.jerk_fixed = torch.linalg.cross(self.omega_vec, self.acc_fixed).contiguous()

        # Force components in moving frame: (thrust, 0, drag)
        force_moving = torch.stack((self.d_t, torch.zeros_like(self.d_t), -self.d_q), dim=-1)
        
        # Rotate forces to fixed frame
        self.force_fixed = torch.einsum('tbik,tbsk->tsbi', rot, force_moving).contiguous()

        # Compute force time derivatives using high-order finite differences
        dt = self.tau[1] - self.tau[0]
        n = force_moving.shape[0]
        df_dt_moving = torch.zeros_like(force_moving)

        # Interior points: 4th-order central difference
        # f'(i) = [-f(i+2) + 8*f(i+1) - 8*f(i-1) + f(i-2)] / (12*dt)
        df_dt_moving[2:-2] = (
            -force_moving[4:] 
            + 8.0 * force_moving[3:-1] 
            - 8.0 * force_moving[1:-3] 
            + force_moving[:-4]
        ) / (12.0 * dt)

        # Boundary: 4th-order forward difference at t=0
        # f'(t0) = [-25/12*f0 + 4*f1 - 3*f2 + 4/3*f3 - 1/4*f4] / dt
        df_dt_moving[0] = (
            -25/12 * force_moving[0] + 4.0 * force_moving[1] - 3.0 * force_moving[2] 
            + 4/3 * force_moving[3] - 0.25 * force_moving[4]
        ) / dt

        # Boundary: 4th-order forward difference at t=t1
        # f'(t1) = [-1/4*f0 - 5/6*f1 + 3/2*f2 - 1/2*f3 + 1/12*f4] / dt
        df_dt_moving[1] = (
            -0.25 * force_moving[0] - 5/6 * force_moving[1] + 1.5 * force_moving[2] 
            - 0.5 * force_moving[3] + 1/12 * force_moving[4]
        ) / dt

        # Boundary: 4th-order backward difference at t=tn
        # f'(tn) = [25/12*fn - 4*fn-1 + 3*fn-2 - 4/3*fn-3 + 1/4*fn-4] / dt
        df_dt_moving[-1] = (
            25/12 * force_moving[-1] - 4.0 * force_moving[-2] + 3.0 * force_moving[-3] 
            - 4/3 * force_moving[-4] + 0.25 * force_moving[-5]
        ) / dt

        # Boundary: 4th-order backward difference at t=tn-1
        # f'(tn-1) = [1/4*fn + 5/6*fn-1 - 3/2*fn-2 + 1/2*fn-3 - 1/12*fn-4] / dt
        df_dt_moving[-2] = (
            0.25 * force_moving[-1] + 5/6 * force_moving[-2] - 1.5 * force_moving[-3] 
            + 0.5 * force_moving[-4] - 1/12 * force_moving[-5]
        ) / dt

        # Rotate force derivatives to fixed frame and add kinematic contribution
        # Total derivative: d(F_fixed)/dt = dF_rot/dt + ω × F_fixed
        df_dt_fixed = torch.einsum('tbik,tbsk->tsbi', rot, df_dt_moving).contiguous()
        self.force_der_fixed = torch.linalg.cross(self.omega_vec, self.force_fixed)
        self.force_der_fixed.add_(df_dt_fixed)

    def get_rf(self, m1: int, m2: int, inv_r: torch.Tensor, inv_omr: torch.Tensor) -> torch.Tensor:
        """Compute combined scaling factors for compact source formulation.
        
        Evaluates r^(-m1) * (1 - M_r)^(-m2) efficiently by fusing inverse computations.
        
        Args:
            m1: Power exponent for inverse distance scaling (0, 1, or 2)
            m2: Power exponent for Mach-number scaling (0, 1, or 2)
            inv_r: Precomputed inverse distances (1/r)
            inv_omr: Precomputed inverse Doppler factor (1 / (1 - M_r))
            
        Returns:
            Combined scaling tensor of shape (time, section, blade, observer)
        """
        # Compute r^(-m1): returns 1, 1/r, or 1/r^2 based on m1
        res_r = inv_r if m1 == 1 else (inv_r.square() if m1 == 2 else torch.ones_like(inv_r))
        # Compute (1 - M_r)^(-m2): returns 1, 1/(1-M_r), or 1/(1-M_r)^2 based on m2
        res_omr = inv_omr if m2 == 1 else (inv_omr.square() if m2 == 2 else torch.ones_like(inv_omr))
        # Multiply the two factors
        return res_r.mul(res_omr)

    def get_rp(
        self,
        m1: int,
        m2: int,
        inv_r: torch.Tensor,
        inv_omr: torch.Tensor,
        m_r: torch.Tensor,
        m_mag_sq: torch.Tensor,
        m_r_dot: torch.Tensor
    ) -> torch.Tensor:
        """Compute derivative of rf term in compact source formulation.
        
        Evaluates the polynomial contribution involving Mach numbers and distance scaling:
        (M_r - M^2) * a_inf * m2 * r^(-(m1+1)) * (1-M_r)^(-(m2+1))
        + m2 * M_r_dot * r^(-m1) * (1-M_r)^(-(m2+1))
        + a_inf * (m1 - m2) * M_r * r^(-(m1+1)) * (1-M_r)^(-m2)
        
        Args:
            m1: Primary distance scaling exponent
            m2: Primary Mach scaling exponent
            inv_r: Precomputed 1/r
            inv_omr: Precomputed 1/(1 - M_r)
            m_r: Mach number component along observer direction
            m_mag_sq: Square of Mach number magnitude
            m_r_dot: Time derivative of M_r
            
        Returns:
            Polynomial term tensor
        """
        # Compute required scaling factors
        rf_m1_m2p1 = self.get_rf(m1, m2 + 1, inv_r, inv_omr)
        rf_m1p1_m2p1 = self.get_rf(m1 + 1, m2 + 1, inv_r, inv_omr)
        rf_m1p1_m2 = self.get_rf(m1 + 1, m2, inv_r, inv_omr)
        
        # First term: (M_r - M^2) * a_inf * m2 * rf(m1+1, m2+1)
        res = (m_r.sub(m_mag_sq)).mul_(self.a_inf).mul_(m2).mul_(rf_m1p1_m2p1)
        # Second term: m2 * M_r_dot * rf(m1, m2+1)
        res.add_(m2 * rf_m1_m2p1 * m_r_dot)
        # Third term: a_inf * (m1 - m2) * M_r * rf(m1+1, m2)
        res.add_(self.a_inf * (m1 - m2) * m_r * rf_m1p1_m2)
        return res

    @torch.no_grad()
    def calculate_f1a_pressure(self, observers: np.ndarray) -> torch.Tensor:
        """Compute F1A pressure at observer locations using compact source formulation.
        
        Calculates acoustic pressure contributions from thickness and loading sources using
        the compact source approximation of the Farassat 1A formulation in Mach-number form.
        
        Args:
            observers: Observer positions (num_obs, 3) with coordinates (x, y, z) in meters
            
        Returns:
            Pressure tensor of shape (time, section, blade, observer, 2) where:
                [..., 0] = thickness source pressure (monopole)
                [..., 1] = loading source pressure (dipole)
        """
        # Convert observer positions to tensor and add batch dimensions
        # Shape after reshaping: (1, 1, 1, num_obs, 3)
        obs = torch.as_tensor(observers, dtype=self.dtype, device=self.device)[None, None, None, :, :]
        
        # Vector from source to observer: r_i = x_obs - x_source
        ri = obs - self.pos_fixed[..., None, :]
        # Distance magnitude: r = ||r_i||
        r = torch.linalg.norm(ri, dim=-1)
        inv_r = r.reciprocal()
        # Unit vector pointing from source to observer: r_hat = r_i / r
        r_hat = ri.mul(inv_r[..., None])

        # Velocity and acceleration vectors (fixed frame)
        v_vec = self.vel_fixed[..., None, :]
        a_vec = self.acc_fixed[..., None, :]
        
        # Mach number vector: M = v / a_inf
        inv_a = self.a_inf.reciprocal()
        m_vec = v_vec.mul(inv_a)
        # Mach number magnitude
        v_mag = torch.linalg.norm(v_vec, dim=-1)
        m_mag_sq = (v_mag.mul(inv_a)).square_()
        # Mach number component along observer direction: M_r = M · r_hat
        m_r = torch.sum(m_vec.mul(r_hat), dim=-1)
        
        # Doppler factor inverse: 1 / (1 - M_r)
        inv_omr = (1.0 - m_r).reciprocal_()

        # Time derivative of Mach magnitude: dM/dt = (v·a)/(a_inf * |v|)
        m_dot = torch.sum(v_vec.mul(a_vec), dim=-1).mul_(self.a_inf.mul(v_mag).add_(1e-8).reciprocal_())
        
        # Time derivative of r_hat: dr_hat/dt = (1/r) * [a_inf * (M - M_r*r_hat) * (-1)]
        # Intermediate term for efficiency
        term_r_hat_dot = (self.a_inf.mul(inv_r)).neg_()
        r_hat_dot = term_r_hat_dot[..., None].mul(m_vec.sub(m_r[..., None].mul(r_hat)))

        # Time derivative of M_r: dM_r/dt = (a·r_hat)/a_inf + a_inf*(M_r^2 - M^2)/r
        m_r_dot = torch.sum(a_vec.mul(r_hat), dim=-1).mul_(inv_a).add_(
            (self.a_inf.mul(inv_r)).mul_(m_r.square().sub_(m_mag_sq))
        )

        # Second time derivative of M_r: d2M_r/dt2
        # Computed from jerk, acceleration changes, and geometric/kinematic terms
        m_r_ddot = torch.sum(self.jerk_fixed[..., None, :].mul(r_hat), dim=-1)
        m_r_ddot.add_(torch.sum(a_vec.mul(r_hat_dot), dim=-1)).mul_(inv_a)
        
        # Geometric term: (M_r^2 - M^2) * (r_hat·v) / r^2
        geo_term = (m_r.square().sub_(m_mag_sq)).mul_(torch.sum(r_hat.mul(v_vec), dim=-1)).mul_(inv_r.square())
        # Kinematic term: (M_r*dM_r/dt - |M|*dM/dt) / r * 2
        kin_term = (m_r.mul(m_r_dot).sub_(m_mag_sq.sqrt().mul(m_dot))).mul_(2.0).mul_(inv_r)
        
        # Add remaining contributions to second derivative
        m_r_ddot.add_(self.a_inf.mul(geo_term.add_(kin_term)))

        # Pre-compute all required scaling factors for F1A formulation
        rf02 = self.get_rf(0, 2, inv_r, inv_omr)
        rf22 = self.get_rf(2, 2, inv_r, inv_omr)
        rf12 = self.get_rf(1, 2, inv_r, inv_omr)
        rf01 = self.get_rf(0, 1, inv_r, inv_omr)
        rf11 = self.get_rf(1, 1, inv_r, inv_omr)
        rf21 = self.get_rf(2, 1, inv_r, inv_omr)
        
        # Pre-compute derivative terms
        rp22 = self.get_rp(2, 2, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)
        rp12 = self.get_rp(1, 2, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)
        rp01 = self.get_rp(0, 1, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)
        rp11 = self.get_rp(1, 1, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)

        # Thickness source scalar coefficient (F1A formulation)
        # c_1A = rf02 * [a_inf * (rp22*(M_r - M^2) + rf22*(...)) + rf12*dM_r/dt + rp12*dM_r/dt2 + rf01*rp01*rp11]
        c1a = (rf02.mul(self.a_inf.mul(rp22.mul(m_r.sub(m_mag_sq)).add_(rf22.mul(m_r_dot.sub(2.0*m_mag_sq.sqrt().mul(inv_a).mul(m_dot)))))))
        c1a.add_(m_r_ddot.mul(rf12)).add_(m_r_dot.mul(rp12))
        c1a.mul_(rf02).add_(rf01.mul(rp01).mul(rp11))

        # Loading source coefficients (F1A formulation)
        # d_1A = rf01 * rf11 * r_hat (direction for first loading term)
        d1a = (rf01.mul(rf11))[..., None].mul(r_hat)
        # e_1A = rf01 * (rp11 * r_hat + rf11 * dr_hat/dt) + a_inf * rf21 * r_hat (direction for second loading term)
        e1a = (rf01[..., None].mul(rp11[..., None].mul(r_hat).add_(rf11[..., None].mul(r_hat_dot))))
        e1a.add_(self.a_inf.mul(rf21[..., None]).mul(r_hat))

        # Thickness source pressure contribution (monopole)
        # p_monopole = (rho/4π) * area * dr * c_1A
        p_m = self.thickness_strength.mul(c1a)
        
        # Loading source pressure contribution (dipole)
        # First term: F · e_1A
        p_d_term1 = torch.sum(self.force_fixed[..., None, :].mul(e1a), dim=-1)
        # Second term: dF/dt · d_1A
        p_d_term2 = torch.sum(self.force_der_fixed[..., None, :].mul(d1a), dim=-1)
        # Total dipole: (dr/4π) * (1/a_inf) * (p_d_term1 + p_d_term2)
        p_d = (p_d_term1.add_(p_d_term2)).mul_(inv_a).mul_(self.dipole_strength)

        # Return both monopole and dipole contributions
        return torch.stack((p_m, p_d), dim=-1)

    @torch.no_grad()
    def get_observer_times(self, observers: np.ndarray) -> torch.Tensor:
        """Compute retarded times for source-observer pairs.
        
        Calculates the time at which acoustic waves reach observers accounting for
        wave propagation delay: t_obs = t_source + r / a_inf
        
        Args:
            observers: Observer positions (num_obs, 3) with coordinates (x, y, z) in meters
            
        Returns:
            Retarded times tensor of shape (time, 1, 1, num_obs) in seconds
        """
        # Convert observer positions to tensor and add batch dimensions
        obs = torch.as_tensor(observers, dtype=self.dtype, device=self.device)[None, None, None, :, :]
        # Compute distance from source to observer
        dist = torch.linalg.norm(obs.sub(self.pos_fixed[..., None, :]), dim=-1)
        # Compute retarded time: tau_obs = tau_source + distance / sound_speed
        return self.tau[:, None, None, None].add(dist.mul(self.a_inf.reciprocal()))