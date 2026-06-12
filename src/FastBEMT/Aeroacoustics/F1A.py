from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from ..Kinematics import Kinematics


class F1A:
    '''Farassat 1A acoustic source formulation.

    Computes thickness (monopole) and loading (dipole) noise from rotating
    propeller blades using compact source approximation. GPU-accelerated via PyTorch.
    '''

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
        device: str,
        kinematics: Kinematics | None = None,
        section_mask: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> None:
        '''Initialize F1A acoustic source array.

        Args:
            rho: Fluid density (kg/m³).
            a_inf: Speed of sound (m/s).
            r: Radial positions (m), shape (n_sections,).
            dr: Radial widths (m), shape (n_sections,).
            area: Section areas (m²), shape (n_sections,).
            chord: Chord lengths (m), shape (n_sections,).
            twist: Twist angles (deg), shape (n_sections,).
            com_shift_forward: Chordwise center offset (m), shape (n_sections,).
            com_shift_up: Normal center offset (m), shape (n_sections,).
            source_times: Emission times (s), shape (n_times,).
            omega: Angular velocity (rad/s).
            d_t: Thrust distribution (N), shape (n_times, n_blades, n_sections).
            d_q: Torque distribution (N·m), shape (n_times, n_blades, n_sections).
            blade_angles: Blade phase offsets (rad), shape (n_blades,).
            device: PyTorch device ('cpu' or 'cuda').
            kinematics: Shared propeller kinematics to reuse for source motion.
            section_mask: Boolean mask selecting the Kinematics sections used
                by the supplied force distributions.
        '''
        torch.backends.opt_einsum.strategy = "branch-all"
        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = torch.float32

        # Ambient flow properties
        self.rho: torch.Tensor = torch.tensor(rho, dtype=self.dtype, device=self.device)
        self.a_inf: torch.Tensor = torch.tensor(
            a_inf, dtype=self.dtype, device=self.device
        )
        self.omega: torch.Tensor = torch.tensor(
            omega, dtype=self.dtype, device=self.device
        )

        # Angular velocity vector (rotates around x-axis)
        self.omega_vec: torch.Tensor = torch.zeros(
            (1, 1, 1, 3), dtype=self.dtype, device=self.device
        )
        self.omega_vec[..., 0] = self.omega

        # Blade geometry arrays
        self.r: torch.Tensor = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        self.dr: torch.Tensor = torch.as_tensor(
            dr, dtype=self.dtype, device=self.device
        )
        self.area: torch.Tensor = torch.as_tensor(
            area, dtype=self.dtype, device=self.device
        )
        self.chord: torch.Tensor = torch.as_tensor(
            chord, dtype=self.dtype, device=self.device
        )
        self.twist_rad: torch.Tensor = torch.deg2rad(
            torch.as_tensor(twist, dtype=self.dtype, device=self.device)
        )

        # Pre-compute thickness source strength: (rho / 4π) * area * dr
        # Shape: (1, Ns, 1, 1) for broadcasting in pressure summation
        thickness_strength: torch.Tensor = (
            (self.rho / (4.0 * np.pi)) * self.area * self.dr
        )
        self.thickness_strength: torch.Tensor = thickness_strength[None, :, None, None]

        # Pre-compute dipole scaling: (dr / 4π)
        # Multiplied by 1/a_inf during pressure calculation for efficiency
        dipole_strength: torch.Tensor = self.dr / (4.0 * np.pi)
        self.dipole_strength: torch.Tensor = dipole_strength[None, :, None, None]

        # Center-of-mass shift parameters
        self.com_shift_forward: torch.Tensor = torch.as_tensor(
            com_shift_forward, dtype=self.dtype, device=self.device
        )
        self.com_shift_up: torch.Tensor = torch.as_tensor(
            com_shift_up, dtype=self.dtype, device=self.device
        )

        # Force distributions
        self.d_t: torch.Tensor = torch.tensor(d_t, dtype=self.dtype, device=self.device)
        self.d_q: torch.Tensor = torch.tensor(d_q, dtype=self.dtype, device=self.device)
        self.tau: torch.Tensor = torch.as_tensor(
            source_times, dtype=self.dtype, device=self.device
        )
        self.blade_angles: torch.Tensor = torch.as_tensor(
            blade_angles, dtype=self.dtype, device=self.device
        )

        # Problem dimensions
        self.nt: int = self.tau.shape[0]
        self.ns: int = self.r.shape[0]
        self.nb: int = self.blade_angles.shape[0]
        self.kinematics = kinematics
        self.section_mask = section_mask

        self._initialize_geometry_and_kinematics()

    def _initialize_geometry_and_kinematics(self) -> None:
        '''Compute kinematic properties of rotating blade sources.

        Calculates position, velocity, acceleration, jerk, and force derivatives
        in the fixed reference frame.
        
        Stores pos_fixed, vel_fixed, acc_fixed, jerk_fixed, force_fixed,
        and force_der_fixed as instance attributes.
        '''
        if self.kinematics is not None:
            self._initialize_from_kinematics()
            return

        # Compute rotation angles at each time step and blade
        angles: torch.Tensor = (
            self.omega * self.tau[:, None] + self.blade_angles[None, :]
        )
        c, s = torch.cos(angles), torch.sin(angles)
        z, o = torch.zeros_like(c), torch.ones_like(c)

        # R_g2b: Global -> Blade (nt, nb, 3, 3)
        R_g2b = torch.stack([
            torch.stack([o, z, z], dim=-1),   # Row 0: Axial
            torch.stack([z, c, s], dim=-1),   # Row 1: Spanwise
            torch.stack([z, -s, c], dim=-1)   # Row 2: Tangential
        ], dim=-2)

        # 2. Setup airfoil rotation (Twist around Y)
        ct, st = torch.cos(self.twist_rad), torch.sin(self.twist_rad)
        zt, ot = torch.zeros_like(ct), torch.ones_like(ct)

        # R_b2a: Blade -> Airfoil (ns, 3, 3) 
        # X_af = Backwards Chord, Z_af = Upward Normal
        R_b2a = torch.stack([
            torch.stack([-st, zt, -ct], dim=-1), # Row 0: New X (TE)
            torch.stack([zt,  ot, zt],  dim=-1), # Row 1: New Y (Span)
            torch.stack([ct,  zt, -st], dim=-1)  # Row 2: New Z (Normal)
        ], dim=-2)

        com_shift_blade_frame = torch.stack([
            self.com_shift_up,                       
            torch.zeros_like(self.r), 
            self.com_shift_forward
        ], dim=-1) # (ns, 3)

        pos_airfoil = torch.stack([
            torch.zeros_like(self.r), # X in blade frame is zero (rotation axis)
            self.r,                    # Y in blade frame is spanwise position
            torch.zeros_like(self.r)  # Z in blade frame is zero (on chord line
        ], dim=-1) # (ns, 3)

        # Transform positions from blade frame to fixed frame using rotation matrices
        # Shape: (time, section, blade, xyz)
        pos_blade = torch.einsum('skj,sj->sk', R_b2a.transpose(-1, -2), pos_airfoil) + com_shift_blade_frame

        self.pos_fixed: torch.Tensor = torch.einsum(
            'tbkj,sj->tsbk', R_g2b.transpose(-1, -2), pos_blade
        ).contiguous()

        # Compute kinematic derivatives by successive cross products with angular velocity
        # v = ω × r, a = ω × v, j = ω × a
        self.vel_fixed: torch.Tensor = torch.linalg.cross(
            self.omega_vec, self.pos_fixed
        ).contiguous()
        self.acc_fixed: torch.Tensor = torch.linalg.cross(
            self.omega_vec, self.vel_fixed
        ).contiguous()
        self.jerk_fixed: torch.Tensor = torch.linalg.cross(
            self.omega_vec, self.acc_fixed
        ).contiguous()

        # Force components in blade-fixed frame: (thrust, 0, drag)
        force_moving: torch.Tensor = torch.stack(
            (self.d_t, torch.zeros_like(self.d_t), -self.d_q), dim=-1
        )

        # Rotate forces to fixed frame
        self.force_fixed: torch.Tensor = torch.einsum(
            'tbkj,tbsj->tsbk', R_g2b.transpose(-1, -2), force_moving
        ).contiguous()

        self.force_der_fixed: torch.Tensor = torch.linalg.cross(
            self.omega_vec, self.force_fixed
        )

    def _initialize_from_kinematics(self) -> None:
        """Reuse shared source motion and initialize F1A force kinematics."""
        kinematics = self.kinematics
        if kinematics is None:
            raise RuntimeError("Kinematics was not provided.")
        if kinematics.device != self.device:
            raise ValueError("F1A and Kinematics must use the same device.")
        if kinematics.nt != self.nt or kinematics.nb != self.nb:
            raise ValueError(
                "F1A source times and blades must match the Kinematics object."
            )

        if self.section_mask is None:
            if kinematics.ns != self.ns:
                raise ValueError(
                    "section_mask is required when F1A uses a section subset."
                )
            selected_sections: slice | torch.Tensor = slice(None)
        else:
            selected_sections = torch.as_tensor(
                self.section_mask,
                device=self.device,
            )
            if (
                selected_sections.dtype != torch.bool
                or selected_sections.ndim != 1
                or selected_sections.shape[0] != kinematics.ns
            ):
                raise ValueError(
                    "section_mask must be a one-dimensional boolean mask "
                    "matching the Kinematics sections."
                )
            if int(selected_sections.sum().item()) != self.ns:
                raise ValueError(
                    "section_mask must select the same number of sections "
                    "provided to F1A."
                )

        self.omega_vec = kinematics.angular_velocity_vector
        self.pos_fixed = kinematics.section_position_global_frame[
            :, selected_sections
        ].contiguous()
        self.vel_fixed = kinematics.section_velocity_global_frame[
            :, selected_sections
        ].contiguous()
        self.acc_fixed = kinematics.section_acceleration_global_frame[
            :, selected_sections
        ].contiguous()
        self.jerk_fixed = kinematics.section_jerk_global_frame[
            :, selected_sections
        ].contiguous()

        force_moving = torch.stack(
            (self.d_t, torch.zeros_like(self.d_t), -self.d_q),
            dim=-1,
        )
        self.force_fixed = torch.einsum(
            "tbij,tbsj->tsbi",
            kinematics.global_to_blade_rotation_matrix.transpose(-1, -2),
            force_moving,
        ).contiguous()
        self.force_der_fixed = torch.linalg.cross(
            self.omega_vec,
            self.force_fixed,
            dim=-1,
        )

    def get_rf(
        self,
        m1: int,
        m2: int,
        inv_r: torch.Tensor,
        inv_omr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined scaling factors for compact source formulation.

        Evaluates $r^{-m_1} (1 - M_r)^{-m_2}$ efficiently by fusing inverse computations.

        Args:
            m1: Power exponent for inverse distance scaling (0, 1, or 2).
            m2: Power exponent for Mach-number scaling (0, 1, or 2).
            inv_r: Precomputed inverse distances (1/r).
                Shape: (time, section, blade, observer).
            inv_omr: Precomputed inverse Doppler factor (1 / (1 - M_r)).
                Shape: (time, section, blade, observer).

        Returns:
            Combined scaling tensor of shape (time, section, blade, observer).
        """
        # Compute r^(-m1): returns 1, 1/r, or 1/r^2 based on m1
        res_r: torch.Tensor = (
            inv_r
            if m1 == 1
            else (inv_r.square() if m1 == 2 else torch.ones_like(inv_r))
        )
        # Compute (1 - M_r)^(-m2): returns 1, 1/(1-M_r), or 1/(1-M_r)^2 based on m2
        res_omr: torch.Tensor = (
            inv_omr
            if m2 == 1
            else (inv_omr.square() if m2 == 2 else torch.ones_like(inv_omr))
        )
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
        m_r_dot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute derivative of rf term in compact source formulation.

        Evaluates the polynomial contribution involving Mach numbers and distance:
        $(M_r - M^2) a_\\infty m_2 r^{-(m_1+1)} (1-M_r)^{-(m_2+1)}$
        $+ m_2 M_{r,t} r^{-m_1} (1-M_r)^{-(m_2+1)}$
        $+ a_\\infty (m_1 - m_2) M_r r^{-(m_1+1)} (1-M_r)^{-m_2}$

        Args:
            m1: Primary distance scaling exponent.
            m2: Primary Mach scaling exponent.
            inv_r: Precomputed 1/r. Shape: (time, section, blade, observer).
            inv_omr: Precomputed 1/(1 - M_r). Shape: (time, section, blade, observer).
            m_r: Mach number component along observer direction.
                Shape: (time, section, blade, observer).
            m_mag_sq: Square of Mach number magnitude.
                Shape: (time, section, blade, observer).
            m_r_dot: Time derivative of M_r.
                Shape: (time, section, blade, observer).

        Returns:
            Polynomial term tensor of shape (time, section, blade, observer).
        """
        # Compute required scaling factors
        rf_m1_m2p1: torch.Tensor = self.get_rf(m1, m2 + 1, inv_r, inv_omr)
        rf_m1p1_m2p1: torch.Tensor = self.get_rf(m1 + 1, m2 + 1, inv_r, inv_omr)
        rf_m1p1_m2: torch.Tensor = self.get_rf(m1 + 1, m2, inv_r, inv_omr)

        # First term: (M_r - M^2) * a_inf * m2 * rf(m1+1, m2+1)
        res: torch.Tensor = (
            (m_r.sub(m_mag_sq)).mul_(self.a_inf).mul_(m2).mul_(rf_m1p1_m2p1)
        )
        # Second term: m2 * M_r_dot * rf(m1, m2+1)
        res.add_(m2 * rf_m1_m2p1 * m_r_dot)
        # Third term: a_inf * (m1 - m2) * M_r * rf(m1+1, m2)
        res.add_(self.a_inf * (m1 - m2) * m_r * rf_m1p1_m2)
        return res

    def calculate_f1a_pressure(self, observers: np.ndarray) -> torch.Tensor:
        """Compute F1A pressure at observer locations using compact source formulation.

        Calculates acoustic pressure contributions from thickness (monopole) and
        loading (dipole) sources using the compact source approximation of the
        Farassat 1A formulation in Mach-number form. All computations are
        GPU-accelerated via PyTorch tensors.

        Args:
            observers: Observer position coordinates in Cartesian space.
                Shape: (n_observers, 3) with coordinates (x, y, z) in meters.

        Returns:
            Pressure tensor of shape (time, section, blade, observer, 2) where:
                [..., 0] = thickness source pressure (monopole contribution)
                [..., 1] = loading source pressure (dipole contribution)
        """
        # Convert observer positions to tensor and add batch dimensions
        # Shape after reshaping: (1, 1, 1, num_obs, 3)
        obs: torch.Tensor = torch.as_tensor(
            observers, dtype=self.dtype, device=self.device
        )[None, None, None, :, :]

        # Vector from source to observer: r_i = x_obs - x_source
        ri: torch.Tensor = obs - self.pos_fixed[..., None, :]
        # Distance magnitude: r = ||r_i||
        r: torch.Tensor = torch.linalg.norm(ri, dim=-1)
        inv_r: torch.Tensor = r.reciprocal()
        # Unit vector pointing from source to observer: r_hat = r_i / r
        r_hat: torch.Tensor = ri.mul(inv_r[..., None])

        # Velocity and acceleration vectors (fixed frame)
        v_vec: torch.Tensor = self.vel_fixed[..., None, :]
        a_vec: torch.Tensor = self.acc_fixed[..., None, :]

        # Mach number vector: M = v / a_inf
        inv_a: torch.Tensor = self.a_inf.reciprocal()
        m_vec: torch.Tensor = v_vec.mul(inv_a)
        # Mach number magnitude
        v_mag: torch.Tensor = torch.linalg.norm(v_vec, dim=-1)
        m_mag_sq: torch.Tensor = (v_mag.mul(inv_a)).square_()
        # Mach number component along observer direction: M_r = M · r_hat
        m_r: torch.Tensor = torch.sum(m_vec.mul(r_hat), dim=-1)

        # Doppler factor inverse: 1 / (1 - M_r)
        inv_omr: torch.Tensor = (1.0 - m_r).reciprocal_()

        # Time derivative of Mach magnitude: dM/dt = (v·a)/(a_inf * |v|)
        m_dot: torch.Tensor = torch.sum(v_vec.mul(a_vec), dim=-1).mul_(
            self.a_inf.mul(v_mag).add_(1e-8).reciprocal_()
        )

        # Time derivative of r_hat: dr_hat/dt = (1/r) * [a_inf * (M - M_r*r_hat) * (-1)]
        # Intermediate term for efficiency
        term_r_hat_dot: torch.Tensor = (self.a_inf.mul(inv_r)).neg_()
        r_hat_dot: torch.Tensor = term_r_hat_dot[..., None].mul(
            m_vec.sub(m_r[..., None].mul(r_hat))
        )

        # Time derivative of M_r: dM_r/dt = (a·r_hat)/a_inf + a_inf*(M_r^2 - M^2)/r
        m_r_dot: torch.Tensor = (
            torch.sum(a_vec.mul(r_hat), dim=-1)
            .mul_(inv_a)
            .add_((self.a_inf.mul(inv_r)).mul_(m_r.square().sub_(m_mag_sq)))
        )

        # Second time derivative of M_r: d2M_r/dt2
        # Computed from jerk, acceleration changes, and geometric/kinematic terms
        m_r_ddot: torch.Tensor = torch.sum(
            self.jerk_fixed[..., None, :].mul(r_hat), dim=-1
        )
        m_r_ddot.add_(torch.sum(a_vec.mul(r_hat_dot), dim=-1)).mul_(inv_a)

        # Geometric term: (M_r^2 - M^2) * (r_hat·v) / r^2
        geo_term: torch.Tensor = (
            (m_r.square().sub_(m_mag_sq))
            .mul_(torch.sum(r_hat.mul(v_vec), dim=-1))
            .mul_(inv_r.square())
        )
        # Kinematic term: (M_r*dM_r/dt - |M|*dM/dt) / r * 2
        kin_term: torch.Tensor = (
            (m_r.mul(m_r_dot).sub_(m_mag_sq.sqrt().mul(m_dot))).mul_(2.0).mul_(inv_r)
        )

        # Add remaining contributions to second derivative
        m_r_ddot.add_(self.a_inf.mul(geo_term.add_(kin_term)))

        # Pre-compute all required scaling factors for F1A formulation
        rf02: torch.Tensor = self.get_rf(0, 2, inv_r, inv_omr)
        rf22: torch.Tensor = self.get_rf(2, 2, inv_r, inv_omr)
        rf12: torch.Tensor = self.get_rf(1, 2, inv_r, inv_omr)
        rf01: torch.Tensor = self.get_rf(0, 1, inv_r, inv_omr)
        rf11: torch.Tensor = self.get_rf(1, 1, inv_r, inv_omr)
        rf21: torch.Tensor = self.get_rf(2, 1, inv_r, inv_omr)

        # Pre-compute derivative terms
        rp22: torch.Tensor = self.get_rp(2, 2, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)
        rp12: torch.Tensor = self.get_rp(1, 2, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)
        rp01: torch.Tensor = self.get_rp(0, 1, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)
        rp11: torch.Tensor = self.get_rp(1, 1, inv_r, inv_omr, m_r, m_mag_sq, m_r_dot)

        # Thickness source scalar coefficient (F1A formulation)
        # c_1A = rf02 * [a_inf * (rp22*(M_r - M^2) + rf22*(...)) + rf12*dM_r/dt
        #               + rp12*dM_r/dt2 + rf01*rp01*rp11]
        c1a: torch.Tensor = rf02.mul(
            self.a_inf.mul(
                rp22.mul(m_r.sub(m_mag_sq)).add_(
                    rf22.mul(m_r_dot.sub(2.0 * m_mag_sq.sqrt().mul(inv_a).mul(m_dot)))
                )
            )
        )
        c1a.add_(m_r_ddot.mul(rf12)).add_(m_r_dot.mul(rp12))
        c1a.mul_(rf02).add_(rf01.mul(rp01).mul(rp11))

        # Loading source direction vectors (F1A formulation)
        # d_1A = rf01 * rf11 * r_hat (direction for first loading term)
        d1a: torch.Tensor = (rf01.mul(rf11))[..., None].mul(r_hat)
        # e_1A = rf01 * (rp11 * r_hat + rf11 * dr_hat/dt)
        #        + a_inf * rf21 * r_hat (direction for second loading term)
        e1a: torch.Tensor = rf01[..., None].mul(
            rp11[..., None].mul(r_hat).add_(rf11[..., None].mul(r_hat_dot))
        )
        e1a.add_(self.a_inf.mul(rf21[..., None]).mul(r_hat))

        # Thickness source pressure contribution (monopole)
        # p_monopole = (rho/4π) * area * dr * c_1A
        p_m: torch.Tensor = self.thickness_strength.mul(c1a)

        # Loading source pressure contribution (dipole)
        # First term: F · e_1A
        p_d_term1: torch.Tensor = torch.sum(
            self.force_fixed[..., None, :].mul(e1a), dim=-1
        )
        # Second term: dF/dt · d_1A
        p_d_term2: torch.Tensor = torch.sum(
            self.force_der_fixed[..., None, :].mul(d1a), dim=-1
        )
        # Total dipole: (dr/4π) * (1/a_inf) * (p_d_term1 + p_d_term2)
        p_d: torch.Tensor = (
            (p_d_term1.add_(p_d_term2)).mul_(inv_a).mul_(self.dipole_strength)
        )

        # Return both monopole and dipole contributions
        return torch.stack((p_m, p_d), dim=-1)
