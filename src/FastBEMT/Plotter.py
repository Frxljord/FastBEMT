import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class Plotter():
    def __init__(self, propeller):
        self.propeller = propeller

    def plot_observer_report(self):
        n_obs = self.propeller.observer_positions.shape[0]
        fig = plt.figure(figsize=(15, 3.5 * n_obs)) 
        gs = gridspec.GridSpec(n_obs, 2, figure=fig)

        fontsize = 10
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        blade_passing_freq = 1 / self.propeller.acoustic_params.blade_passing_period

        ax_spl = fig.add_subplot(gs[:, 1])

        for idx in range(n_obs):
            ax_time = fig.add_subplot(gs[idx, 0])
            color = colors[idx % len(colors)]
            
            ax_time.plot(self.propeller.t[:, idx], self.propeller.p_m[:, idx], 
                        label='Monopole', linestyle='-', marker='o', markersize=1.5, alpha=0.4)
            ax_time.plot(self.propeller.t[:, idx], self.propeller.p_d[:, idx], 
                        label='Dipole', linestyle='-', marker='s', markersize=1.5, alpha=0.4)
            ax_time.plot(self.propeller.t[:, idx], self.propeller.p_tot[:, idx], 
                        color=color, linestyle='-', marker='d', markersize=2, 
                        linewidth=1.2, label='Total Pressure')
            
            ax_time.set_title(f'Observer {idx} Time History', fontsize=fontsize, fontweight='bold')
            ax_time.set_xlabel('Time [s]', fontsize=fontsize)
            ax_time.set_ylabel('Pressure [Pa]', fontsize=fontsize)
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(loc='upper right', fontsize='small')

            label_text = f'Obs {idx} (OASPL: {self.propeller.oaspl[idx]:.3f} dB)'
            
            ax_spl.semilogx(self.propeller.freq[1:, idx], self.propeller.spl[1:, idx], 
                            color=color, linestyle='-', marker='.', markersize=2, 
                            label=label_text, alpha=0.8)

        ax_spl.set_title('Acoustic Spectrum (SPL)', fontsize=fontsize, fontweight='bold')
        ax_spl.set_xlabel('Frequency [Hz]', fontsize=fontsize)
        ax_spl.set_ylabel('SPL [dB]', fontsize=fontsize)

        max_f = np.max(self.propeller.freq)
        harmonic_freqs = np.arange(blade_passing_freq, max_f, blade_passing_freq)
        for bpf in harmonic_freqs:
            ax_spl.axvline(bpf, color='red', linestyle='--', linewidth=0.8, alpha=0.2)

        ax_spl.grid(True, which='both', alpha=0.3)
        ax_spl.legend(fontsize=fontsize, loc='lower left')
        ax_spl.set_ylim()
        plt.tight_layout()
        plt.show()