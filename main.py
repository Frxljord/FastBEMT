import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyfar as pf
import torch
from FastBEMT.Propeller import Propeller
from FastBEMT.JobParameters import LowFidelityParameters
from FastBEMT.TorchBPM import BPM


def main() -> None:
    """Main analysis pipeline for propeller BEMT and acoustic modeling."""

    with open("Datasets/Propellers/10x7E.pkl", "rb") as f:
        blade_dict = pickle.load(f)

    params = LowFidelityParameters(
        rpm=5000,
        a_inf=343,
        rho=1.225,
        mu=1.81e-5,
        n_blades=blade_dict['n_blades'],
        p_ref=2e-5,
        revolutions=100,
        num_obs_times_per_rev=100,
        device='cuda'
    )

    # Observer positions in meters [x, y, z]
    r_observer = np.array([[0.1, 1.8, 0], [0.1, 0, 1.8]])
    # r_observer = np.array([[0, 1.8, 0]])

    propeller = Propeller(
        propeller_geometry=blade_dict,
        params=params,
        dtype=torch.float32,
    )

    propeller.run_bemt(v_inf=0)
    propeller.run_aeroacoustics(observer_positions=r_observer)
    
    # # Performance output
    # thrust, torque, ct, cp = propeller.compute_total_forces()
    # print(f"BEMT Results: Thrust={thrust:.4f} N, Torque={torque:.4f} N·m, Ct={ct:.6f}, Cp={cp:.6f}")

    # n_obs = r_observer.shape[0]
    # fig = plt.figure(figsize=(15, 3.5 * n_obs))
    # gs = gridspec.GridSpec(n_obs, 2, figure=fig)

    # fontsize = 10
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # blade_passing_freq = 1 / params.blade_passing_period

    # ax_spl = fig.add_subplot(gs[:, 1])

    # for idx in range(n_obs):
    #     ax_time = fig.add_subplot(gs[idx, 0])
    #     color = colors[idx % len(colors)]
        
    #     ax_time.plot(propeller.t[:, idx], propeller.p_m[:, idx], 
    #                 label='Monopole', linestyle='-', marker='o', markersize=1.5, alpha=0.4)
    #     ax_time.plot(propeller.t[:, idx], propeller.p_d[:, idx], 
    #                 label='Dipole', linestyle='-', marker='s', markersize=1.5, alpha=0.4)
    #     ax_time.plot(propeller.t[:, idx], propeller.p_tot[:, idx], 
    #                 color=color, linestyle='-', marker='d', markersize=2, 
    #                 linewidth=1.2, label='Total Pressure')
        
    #     ax_time.set_title(f'Observer {idx} Time History', fontsize=fontsize, fontweight='bold')
    #     ax_time.set_xlabel('Time [s]', fontsize=fontsize)
    #     ax_time.set_ylabel('Pressure [Pa]', fontsize=fontsize)
    #     ax_time.grid(True, alpha=0.3)
    #     ax_time.legend(loc='upper right', fontsize='small')

    #     label_text = f'Obs {idx} (OASPL: {propeller.oaspl[idx]:.3f} dB)'
        
    #     ax_spl.semilogx(propeller.freq[1:, idx], propeller.spl[1:, idx], 
    #                     color=color, linestyle='-', marker='.', markersize=2, 
    #                     label=label_text, alpha=0.8)

    # ax_spl.set_title('Acoustic Spectrum (SPL)', fontsize=fontsize, fontweight='bold')
    # ax_spl.set_xlabel('Frequency [Hz]', fontsize=fontsize)
    # ax_spl.set_ylabel('SPL [dB]', fontsize=fontsize)

    # max_f = np.max(propeller.freq)
    # harmonic_freqs = np.arange(blade_passing_freq, max_f, blade_passing_freq)
    # for bpf in harmonic_freqs:
    #     ax_spl.axvline(bpf, color='red', linestyle='--', linewidth=0.8, alpha=0.2)

    # ax_spl.grid(True, which='both', alpha=0.3)
    # ax_spl.legend(fontsize=fontsize, loc='lower left')
    # ax_spl.set_xlim(10, 10000)
    # ax_spl.set_ylim(-50, 70)
    # plt.tight_layout()
    # plt.show()

    # f1a_spl_3oct = []
    # for fc in third_octave_freqs:
    #     f_low, f_high = fc / (2 ** (1 / 6)), fc * (2 ** (1 / 6))
    #     mask = (propeller.freq >= f_low) & (propeller.freq < f_high)
    #     p_band = np.sum(10 ** (propeller.spl*mask / 10), axis=0)
    #     f1a_spl_3oct.append(10 * np.log10(p_band))

    # f1a_spl_3oct = np.array(f1a_spl_3oct)

    # l_total = 10 * np.log10(
    #     10 ** (f1a_spl_3oct / 10) # ok
    #     + 10 ** (bpm.spl_lbl / 10) # ok
    #     + 10 ** (bpm.spl_tbl / 10)
    #     + 10 ** (bpm.spl_teb / 10)
    #     + 10 ** (bpm.spl_ti / 10)
    #     + 10 ** (bpm.spl_tip / 10) # ok
    # )

    # ospl_third_octave = 10 * np.log10(np.sum(10 ** (l_total / 10), axis=0))
    # print(f"F1A Results: OSPL={ospl_third_octave} dB")

    plt.figure(figsize=(10, 6))
    f = propeller.third_octave_freqs.cpu().numpy()

    plt.semilogx(f, propeller.spl_breakdown['tbl'][:, 0].cpu().numpy(), 'b--', label='TBL', alpha=0.7)
    plt.semilogx(f, propeller.spl_breakdown['lbl'][:, 0].cpu().numpy(), 'g--', label='LBL', alpha=0.7)
    plt.semilogx(f, propeller.spl_breakdown['teb'][:, 0].cpu().numpy(), 'y--', label='TEB', alpha=0.7)
    plt.semilogx(f, propeller.spl_breakdown['ti'][:, 0].cpu().numpy(), 'm--', label='TI', alpha=0.7)
    plt.semilogx(f, propeller.spl_breakdown['tv'][:, 0].cpu().numpy(), 'r--', label='TV', alpha=0.7)
    plt.semilogx(f, propeller.spl_total[:, 0].cpu().numpy(), 'k-', lw=2, label='Total')

    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('SPL [dB]')
    plt.title('BPM Broadband Noise Breakdown')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()