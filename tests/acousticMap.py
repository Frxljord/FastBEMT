import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.Propeller import Propeller
from src.JobParameters import AerodynamicParameters, AcousticParameters

def main():
    with open("10x7E.pkl", "rb") as f:
        blade_dict = pickle.load(f)

    aerodynamic_params = AerodynamicParameters(
        prop_radius=blade_dict['tip_radius'],
        hub_radius=blade_dict['hub_radius'],
        n_blades=blade_dict['n_blades'],
        rpm=7000, v_inf=0, a_inf=343, rho=1.225, mu=1.81e-5, p_ref=2e-5
    )

    acoustic_params = AcousticParameters(
        aero_params=aerodynamic_params,
        p_ref=2e-5, revolutions=5, num_obs_times_per_rev=72
    )

    res = 50 
    x_range = np.linspace(-2.0, 2.0, res)
    y_range = np.linspace(-2.0, 2.0, res)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X) 

    r_observers = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    propeller = Propeller(
        propeller_geometry=blade_dict,
        aero_params=aerodynamic_params,
        acoustic_params=acoustic_params
    )
    
    propeller.run_bemt()
    propeller.run_compact_f1a(observer_positions=r_observers)

    ospl_grid = propeller.ospl.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    smooth_levels = np.linspace(ospl_grid.min(), ospl_grid.max(), 100)
    cf = ax.contourf(Y, X, ospl_grid, levels=smooth_levels, cmap='magma', extend='both')
    
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('OSPL [dB]', fontweight='bold')

    line_levels = np.arange(0, np.max(ospl_grid) + 10, 10)
    contours = ax.contour(Y, X, ospl_grid, levels=line_levels, colors='white', linewidths=0.8, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f dB')

    ax.plot(0, 0, 'ro', markersize=8, label='Propeller Center', markeredgecolor='white')
    
    ax.set_title('OSPL Map', fontweight='bold', fontsize=12)
    ax.set_xlabel('Horizontal Position (Y) [m]', fontsize=10)
    ax.set_ylabel('Vertical Position (X) [m]', fontsize=10)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()