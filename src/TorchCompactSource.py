import torch
import numpy as np

class TorchCompactAcousticSourceArray:
    def __init__(
        self, 
        rho: float, a_inf: float, 
        r, dr, area, chord, twist, 
        com_shift_forward, com_shift_up, 
        source_times, omega: float, 
        d_t, d_q, blade_angles,
        device: str = "cuda"
    ):
        torch.backends.opt_einsum.strategy = 'branch-all'
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.rho = torch.tensor(rho, dtype=self.dtype, device=self.device)
        self.a_inf = torch.tensor(a_inf, dtype=self.dtype, device=self.device)
        self.omega = torch.tensor(omega, dtype=self.dtype, device=self.device)

        self.omega_vec = torch.zeros((1, 1, 1, 3), dtype=self.dtype, device=self.device)
        self.omega_vec[..., 0] = self.omega

        self.r = torch.as_tensor(r, dtype=self.dtype, device=self.device)
        self.dr = torch.as_tensor(dr, dtype=self.dtype, device=self.device)
        self.area = torch.as_tensor(area, dtype=self.dtype, device=self.device)
        self.chord = torch.as_tensor(chord, dtype=self.dtype, device=self.device)
        self.twist_rad = torch.deg2rad(torch.as_tensor(twist, dtype=self.dtype, device=self.device))
        
        # Grid Constant Pre-calculation
        # Pre-calculating the thickness source strength: (rho / 4pi) * area * dr
        # Shape: (1, Ns, 1, 1) for easy broadcasting in pressure summation
        thickness_strength = (self.rho / (4.0 * np.pi)) * self.area * self.dr
        self.thickness_strength = thickness_strength[None, :, None, None]
        
        # Pre-calculating the dipole scaling: (dr / 4pi)
        # We will multiply by inv_a inside the call as a_inf is scalar
        dipole_strength = self.dr / (4.0 * np.pi)
        self.dipole_strength = dipole_strength[None, :, None, None]

        self.com_shift_forward = torch.as_tensor(com_shift_forward, dtype=self.dtype, device=self.device)
        self.com_shift_up = torch.as_tensor(com_shift_up, dtype=self.dtype, device=self.device)

        self.d_t = torch.tensor(d_t, dtype=self.dtype, device=self.device)
        self.d_q = torch.tensor(d_q, dtype=self.dtype, device=self.device)
        self.tau = torch.as_tensor(source_times, dtype=self.dtype, device=self.device)
        self.blade_angles = torch.as_tensor(blade_angles, dtype=self.dtype, device=self.device)

        self.nt, self.ns, self.nb = self.tau.shape[0], self.r.shape[0], self.blade_angles.shape[0]
        
        self._initialize_geometry_and_kinematics()

    @torch.no_grad()
    def _initialize_geometry_and_kinematics(self):
        pos_moving = torch.stack([
            self.chord * self.com_shift_up * torch.sin(self.twist_rad),
            self.r,
            self.chord * self.com_shift_forward * torch.sin(self.twist_rad)
        ], dim=-1)

        angles = self.omega * self.tau[:, None] + self.blade_angles[None, :]
        c, s = torch.cos(angles), torch.sin(angles)
        
        rot = torch.zeros((self.nt, self.nb, 3, 3), dtype=self.dtype, device=self.device)
        rot[..., 0, 0] = 1.0
        rot[..., 1, 1], rot[..., 1, 2] = c, -s
        rot[..., 2, 1], rot[..., 2, 2] = s, c

        # Contiguous ensures memory layout is optimal for the following einsum/cross
        self.pos_fixed = torch.einsum('tbik,sk->tsbi', rot, pos_moving).contiguous()

        self.vel_fixed = torch.linalg.cross(self.omega_vec, self.pos_fixed).contiguous()
        self.acc_fixed = torch.linalg.cross(self.omega_vec, self.vel_fixed).contiguous()
        self.jerk_fixed = torch.linalg.cross(self.omega_vec, self.acc_fixed).contiguous()

        force_moving = torch.stack((self.d_t, torch.zeros_like(self.d_t), -self.d_q), dim=-1)
        self.force_fixed = torch.einsum('tbik,tbsk->tsbi', rot, force_moving).contiguous()

        dt = self.tau[1] - self.tau[0] 
        df_dt_moving = torch.zeros_like(force_moving)
        df_dt_moving[1:-1] = (force_moving[2:] - force_moving[:-2]) / (2.0 * dt)
        
        # 2. Forward difference at the start (O(dt^2))
        # f'(t0) ≈ [-3f(t0) + 4f(t1) - f(t2)] / (2*dt)
        df_dt_moving[0] = (-3.0 * force_moving[0] + 4.0 * force_moving[1] - force_moving[2]) / (2.0 * dt)
        
        # 3. Backward difference at the end (O(dt^2))
        # f'(tn) ≈ [3f(tn) - 4f(tn-1) + f(tn-2)] / (2*dt)
        df_dt_moving[-1] = (3.0 * force_moving[-1] - 4.0 * force_moving[-2] + force_moving[-3]) / (2.0 * dt)
        
        df_dt_fixed = torch.einsum('tbik,tbsk->tsbi', rot, df_dt_moving).contiguous()
        
        # In-place addition for the force derivative
        self.force_der_fixed = torch.linalg.cross(self.omega_vec, self.force_fixed)
        self.force_der_fixed.add_(df_dt_fixed)

    @torch.no_grad()
    def calculate_f1a_pressure(self, observers: np.ndarray) -> torch.Tensor:
        obs = torch.as_tensor(observers, dtype=self.dtype, device=self.device)[None, None, None, :, :]
        
        ri = obs - self.pos_fixed[..., None, :]
        r = torch.linalg.norm(ri, dim=-1)
        inv_r = r.reciprocal()
        r_hat = ri.mul(inv_r[..., None]) # In-place mul

        v_vec = self.vel_fixed[..., None, :]
        a_vec = self.acc_fixed[..., None, :]
        
        inv_a = self.a_inf.reciprocal()
        m_vec = v_vec.mul(inv_a) # In-place mul
        v_mag = torch.linalg.norm(v_vec, dim=-1)
        m_mag_sq = (v_mag.mul(inv_a)).square_() # In-place mul and square
        m_r = torch.sum(m_vec.mul(r_hat), dim=-1)
        
        inv_omr = (1.0 - m_r).reciprocal_() # In-place reciprocal

        def get_rf_fused(m1, m2): 
            res_r = inv_r if m1 == 1 else (inv_r.square() if m1 == 2 else torch.ones_like(inv_r))
            res_omr = inv_omr if m2 == 1 else (inv_omr.square() if m2 == 2 else torch.ones_like(inv_omr))
            return res_r.mul(res_omr)

        m_dot = torch.sum(v_vec.mul(a_vec), dim=-1).mul_(self.a_inf.mul(v_mag).add_(1e-8).reciprocal_())
        
        # Intermediate term for r_hat_dot
        term_r_hat_dot = (self.a_inf.mul(inv_r)).neg_()
        r_hat_dot = term_r_hat_dot[..., None].mul(m_vec.sub(m_r[..., None].mul(r_hat)))

        m_r_dot = torch.sum(a_vec.mul(r_hat), dim=-1).mul_(inv_a).add_(
            (self.a_inf.mul(inv_r)).mul_(m_r.square().sub_(m_mag_sq))
        )

        def get_rp_fused(m1, m2):
            rf_m1_m2p1 = get_rf_fused(m1, m2 + 1)
            rf_m1p1_m2p1 = get_rf_fused(m1 + 1, m2 + 1)
            rf_m1p1_m2 = get_rf_fused(m1 + 1, m2)
            
            # Use in-place ops for the polynomial term
            res = (m_r.sub(m_mag_sq)).mul_(self.a_inf).mul_(m2).mul_(rf_m1p1_m2p1)
            res.add_(m2 * rf_m1_m2p1 * m_r_dot)
            res.add_(self.a_inf * (m1 - m2) * m_r * rf_m1p1_m2)
            return res

        # m_r_ddot involves many terms; we carefully use in-place where possible
        m_r_ddot = torch.sum(self.jerk_fixed[..., None, :].mul(r_hat), dim=-1)
        m_r_ddot.add_(torch.sum(a_vec.mul(r_hat_dot), dim=-1)).mul_(inv_a)
        
        geo_term = (m_r.square().sub_(m_mag_sq)).mul_(torch.sum(r_hat.mul(v_vec), dim=-1)).mul_(inv_r.square())
        kin_term = (m_r.mul(m_r_dot).sub_(m_mag_sq.sqrt().mul(m_dot))).mul_(2.0).mul_(inv_r)
        
        m_r_ddot.add_(self.a_inf.mul(geo_term.add_(kin_term)))

        rf02, rf22, rf12 = get_rf_fused(0, 2), get_rf_fused(2, 2), get_rf_fused(1, 2)
        rf01, rf11, rf21 = get_rf_fused(0, 1), get_rf_fused(1, 1), get_rf_fused(2, 1)
        rp22, rp12, rp01, rp11 = get_rp_fused(2, 2), get_rp_fused(1, 2), get_rp_fused(0, 1), get_rp_fused(1, 1)

        # Thickness pressure integrand
        c1a = (rf02.mul(self.a_inf.mul(rp22.mul(m_r.sub(m_mag_sq)).add_(rf22.mul(m_r_dot.sub(2.0*m_mag_sq.sqrt().mul(inv_a).mul(m_dot)))))))
        c1a.add_(m_r_ddot.mul(rf12)).add_(m_r_dot.mul(rp12))
        c1a.mul_(rf02).add_(rf01.mul(rp01).mul(rp11))

        # Dipole integrands
        d1a = (rf01.mul(rf11))[..., None].mul(r_hat)
        e1a = (rf01[..., None].mul(rp11[..., None].mul(r_hat).add_(rf11[..., None].mul(r_hat_dot))))
        e1a.add_(self.a_inf.mul(rf21[..., None]).mul(r_hat))

        # Final sums using pre-calculated grid constants
        p_m = self.thickness_strength.mul(c1a)
        
        # Dipole components summed in-place
        p_d_term1 = torch.sum(self.force_fixed[..., None, :].mul(e1a), dim=-1)
        p_d_term2 = torch.sum(self.force_der_fixed[..., None, :].mul(d1a), dim=-1)
        p_d = (p_d_term1.add_(p_d_term2)).mul_(inv_a).mul_(self.dipole_strength)

        return torch.stack((p_m, p_d), dim=-1)

    @torch.no_grad()
    def get_observer_times(self, observers: np.ndarray) -> torch.Tensor:
        obs = torch.as_tensor(observers, dtype=self.dtype, device=self.device)[None, None, None, :, :]
        # Ensure pos_fixed is used contiguously for distance calc
        dist = torch.linalg.norm(obs.sub(self.pos_fixed[..., None, :]), dim=-1)
        return self.tau[:, None, None, None].add(dist.mul(self.a_inf.reciprocal()))