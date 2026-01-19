import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.Propeller import Propeller
from src.JobParameters import AerodynamicParameters, AcousticParameters

def main():
    with open("Datasets/Propellers/10x7E.pkl", "rb") as f:
        blade_dict = pickle.load(f)

    aerodynamic_params = AerodynamicParameters(
        prop_radius=blade_dict['tip_radius'],
        hub_radius=blade_dict['hub_radius'],
        n_blades=blade_dict['n_blades'],
        rpm=7000,
        a_inf=343,
        rho=1.225,
        mu=1.81e-5,
    )

    acoustic_params = AcousticParameters(
        aero_params=aerodynamic_params,
        p_ref=2e-5,
        revolutions=5,
        num_obs_times_per_rev=100
    )

    # Observer positions in meters [x, y, z]
    r_observer = np.array([[0, 1.8, 0],
                           [0, 0, 1.8]])

    propeller = Propeller(
        propeller_geometry=blade_dict,
        aero_params=aerodynamic_params,
        acoustic_params=acoustic_params
    )

    propeller.run_bemt(v_inf=0*np.ones(len(blade_dict['r'])))
    propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=True)

    # Performance output
    thrust, torque, ct, cp = propeller.compute_total_forces()
    print(f"BEMT Results: Thrust={thrust:.4f} N, Torque={torque:.4f} N·m, Ct={ct:.6f}, Cp={cp:.6f}")

    n_obs = r_observer.shape[0]
    fig = plt.figure(figsize=(15, 3.5 * n_obs)) 
    gs = gridspec.GridSpec(n_obs, 2, figure=fig)

    fontsize = 10
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    blade_passing_freq = 1 / acoustic_params.blade_passing_period

    ax_spl = fig.add_subplot(gs[:, 1])

    for idx in range(n_obs):
        ax_time = fig.add_subplot(gs[idx, 0])
        color = colors[idx % len(colors)]
        
        ax_time.plot(propeller.t[:, idx], propeller.p_m[:, idx], 
                    label='Monopole', linestyle='-', marker='o', markersize=1.5, alpha=0.4)
        ax_time.plot(propeller.t[:, idx], propeller.p_d[:, idx], 
                    label='Dipole', linestyle='-', marker='s', markersize=1.5, alpha=0.4)
        ax_time.plot(propeller.t[:, idx], propeller.p_tot[:, idx], 
                    color=color, linestyle='-', marker='d', markersize=2, 
                    linewidth=1.2, label='Total Pressure')
        
        ax_time.set_title(f'Observer {idx} Time History', fontsize=fontsize, fontweight='bold')
        ax_time.set_xlabel('Time [s]', fontsize=fontsize)
        ax_time.set_ylabel('Pressure [Pa]', fontsize=fontsize)
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(loc='upper right', fontsize='small')

        label_text = f'Obs {idx} (OASPL: {propeller.oaspl[idx]:.3f} dB)'
        
        ax_spl.semilogx(propeller.freq[1:, idx], propeller.spl[1:, idx], 
                        color=color, linestyle='-', marker='.', markersize=2, 
                        label=label_text, alpha=0.8)

    ax_spl.set_title('Acoustic Spectrum (SPL)', fontsize=fontsize, fontweight='bold')
    ax_spl.set_xlabel('Frequency [Hz]', fontsize=fontsize)
    ax_spl.set_ylabel('SPL [dB]', fontsize=fontsize)

    max_f = np.max(propeller.freq)
    harmonic_freqs = np.arange(blade_passing_freq, max_f, blade_passing_freq)
    for bpf in harmonic_freqs:
        ax_spl.axvline(bpf, color='red', linestyle='--', linewidth=0.8, alpha=0.2)

    ax_spl.grid(True, which='both', alpha=0.3)
    ax_spl.legend(fontsize=fontsize, loc='lower left')
    ax_spl.set_xlim(10, 10000)
    ax_spl.set_ylim(-50,70)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()