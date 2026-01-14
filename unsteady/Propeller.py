import numpy as np
import pandas as pd
import aerosandbox as asb
from typing import List, Tuple 
from JobParameters import *
from Section import SectionForces

class Propeller:
    """High-level propeller BEMT controller with unsteady Øye inflow."""

    def __init__(self, propeller_geometry: dict, aero_params: AerodynamicParameters, acoustic_params: AcousticParameters):
        self.geometry = propeller_geometry
        self.aero_params = aero_params
        self.acoustic_params = acoustic_params
        
        self.solution_data = pd.DataFrame()
        self._section_airfoils = [asb.Airfoil(coordinates=af) for af in self.geometry["airfoil"]]

        self.sections = []
        for i in range(len(self.geometry["r"])):
            sec = SectionForces(
                airfoil=self._section_airfoils[i],
                r=self.geometry["r"][i],
                dr=self.geometry["dr"][i],
                chord=self.geometry["chord"][i],
                theta=self.geometry["twist"][i],
                v_inf=self.aero_params.v_inf[i],
                propellerParams=self.aero_params
            )
            self.sections.append(sec)

    def run_unsteady_simulation(self, u_box, Lx, Ly, Lz, dt, duration, V_mean):
        """
        Drives the unsteady simulation. 
        Matches steady BEMT by including quasi-steady tangential induction (a_prime).
        """
        nx, ny, nz = u_box.shape
        dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
        steps = int(duration / dt)
        
        history = []
        azimuth_base = 0.0 
        
        # Ensure V_mean is synchronized
        for sec in self.sections:
            sec.v_inf = V_mean

        for step in range(steps):
            t = step * dt
            ix = int((V_mean * t) / dx) % nx
            azimuth_base += self.aero_params.omega * dt
            
            total_T_step = 0
            total_Q_step = 0
            
            # Radial Sweep
            for sec in self.sections:
                dTs_ann = []
                Ws_ann = []
                phis_ann = []
                cDp_ann = [] # Disk-plane drag coefficient for a_prime
                
                # Blade Sweep
                for b in range(self.aero_params.n_blades):
                    beta = azimuth_base + (2 * np.pi * b / self.aero_params.n_blades)
                    
                    # Spatial sampling (Disk centered in turbulence box)
                    y_pos = sec.r * np.cos(beta)
                    z_pos = sec.r * np.sin(beta)
                    iy = int((y_pos + Ly/2) / dy) % ny
                    iz = int((z_pos + Lz/2) / dz) % nz
                    
                    v_wind = V_mean + u_box[ix, iy, iz]
                    
                    # 1. Calculate Forces 
                    # Now returns cD_p to solve for tangential induction
                    dT, dQ, W, phi, cD_p = sec.step_forces(v_wind, beta)
                    
                    dTs_ann.append(dT)
                    Ws_ann.append(W)
                    phis_ann.append(phi)
                    cDp_ann.append(cD_p)
                    
                    total_T_step += dT
                    total_Q_step += dQ
                
                # 2. Update the Wake State
                # Passing the average cD_p allows a_prime to match steady BEMT
                sec.update_oye_state(
                    dt = dt,
                    dT_annulus = sum(dTs_ann),
                    cDp_avg = np.mean(cDp_ann),
                    W_avg = np.mean(Ws_ann),
                    phi_avg = np.mean(phis_ann)
                )

            # 3. Log results
            if step % 10 == 0: # Reduce print frequency for speed
                print(f"Time: {t:.4f}s | Thrust: {total_T_step:.3f}N | Torque: {total_Q_step:.4f}Nm")
                
            history.append({
                'time': t, 
                'thrust': total_T_step, 
                'torque': total_Q_step,
                'v_inf_mean': V_mean
            })

        self.solution_data = pd.DataFrame(history)
        return self.solution_data