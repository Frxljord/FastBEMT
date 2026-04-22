import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple
from .Stress import BladeStressCalculator

if TYPE_CHECKING:
    from ..Propeller import Propeller


class Plotter:
    '''Visualization utilities for propeller acoustic results.'''
    
    def __init__(self, propeller: 'Propeller') -> None:
        '''Initialize plotter with propeller instance.
        
        Args:
            propeller: Propeller object with computed acoustic results.
        '''
        self.propeller = propeller

    def plot_observer_report(self) -> None:
        '''Plot time histories and frequency spectra for all observers.
        
        Creates multi-panel figure showing:
        - Time history of monopole, dipole, and total pressure
        - Frequency spectrum (SPL) with blade passing frequency harmonics
        - Overall A-weighted SPL for each observer
        '''
        n_obs = self.propeller.observer_positions.shape[0]
        fig = plt.figure(figsize=(15, 3.5 * n_obs))
        gs = gridspec.GridSpec(n_obs, 2, figure=fig)

        fontsize = 10
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        blade_passing_freq = 1 / self.propeller.params.blade_passing_period

        ax_spl = fig.add_subplot(gs[:, 1])

        for idx in range(n_obs):
            ax_time = fig.add_subplot(gs[idx, 0])
            color = colors[idx % len(colors)]

            ax_time.plot(
                self.propeller.t[:, idx],
                self.propeller.p_m[:, idx],
                label="Monopole",
                linestyle="-",
                marker="o",
                markersize=1.5,
                alpha=0.4,
            )
            ax_time.plot(
                self.propeller.t[:, idx],
                self.propeller.p_d[:, idx],
                label="Dipole",
                linestyle="-",
                marker="s",
                markersize=1.5,
                alpha=0.4,
            )
            ax_time.plot(
                self.propeller.t[:, idx],
                self.propeller.p_tot[:, idx],
                color=color,
                linestyle="-",
                marker="d",
                markersize=2,
                linewidth=1.2,
                label="Total Pressure",
            )

            ax_time.set_title(
                f"Observer {idx} Time History", fontsize=fontsize, fontweight="bold"
            )
            ax_time.set_xlabel("Time [s]", fontsize=fontsize)
            ax_time.set_ylabel("Pressure [Pa]", fontsize=fontsize)
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(loc="upper right", fontsize="small")

            label_text = f"Obs {idx} (OASPL: {self.propeller.oaspl[idx]:.3f} dB)"

            ax_spl.semilogx(
                self.propeller.freq[1:, idx],
                self.propeller.spl[1:, idx],
                color=color,
                linestyle="-",
                marker=".",
                markersize=2,
                label=label_text,
                alpha=0.8,
            )

        ax_spl.set_title(
            "Acoustic Spectrum (SPL)", fontsize=fontsize, fontweight="bold"
        )
        ax_spl.set_xlabel("Frequency [Hz]", fontsize=fontsize)
        ax_spl.set_ylabel("SPL [dB]", fontsize=fontsize)

        max_f = np.max(self.propeller.freq)
        harmonic_freqs = np.arange(blade_passing_freq, max_f, blade_passing_freq)
        for bpf in harmonic_freqs:
            ax_spl.axvline(bpf, color="red", linestyle="--", linewidth=0.8, alpha=0.2)

        ax_spl.grid(True, which="both", alpha=0.3)
        ax_spl.legend(fontsize=fontsize, loc="lower left")
        ax_spl.set_ylim()
        plt.tight_layout()
        plt.show()

    def plot_acoustic_map(
        self,
        grid_size: int = 26,
        domain_size: float = 5.0,
        figsize: Tuple[float, float] = (4.5, 2),
        cmap: str = 'magma',
        contour_levels: Optional[Tuple[float]] = None,
        mirror: Optional[bool] = True,
        save_path: Optional[str] = None,
    ) -> None:
        '''Plot 2D acoustic radiation pattern as contour map.
        
        Visualizes pre-calculated OASPL data. Assumes propeller has already been
        run through run_aeroacoustics() and postprocess().
        
        Args:
            grid_size: Number of grid points in each direction (default 26).
            domain_size: Size of domain in meters (default 5.0).
            figsize: Figure size tuple (width, height) in inches.
            cmap: Colormap name (default 'magma').
            show_contours: Whether to show contour lines (default True).
            save_path: Optional path to save figure as PDF.
        '''
        # Reshape and mirror pre-calculated data
        third_octave_oaspl = self.propeller.third_octave_total_oaspl.reshape(grid_size, grid_size)
        if mirror:
            third_octave_oaspl_full = np.vstack([
                third_octave_oaspl[::-1, :],     # Flip for negative y
                third_octave_oaspl[1:, :],       # Original positive y, skip y=0 to avoid duplication
            ])
        else:
            third_octave_oaspl_full = third_octave_oaspl

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        vmin = np.floor(third_octave_oaspl_full.min() / 10) * 10
        vmax = np.ceil(third_octave_oaspl_full.max() / 10) * 10
        
        # Create full grid for plotting
        x_range_plot = np.linspace(-domain_size, domain_size, grid_size)
        y_range_plot = np.linspace(-domain_size, domain_size, 2 * grid_size - 1)
        X_plot, Y_plot = np.meshgrid(x_range_plot, y_range_plot)

        # Plot with smooth coloring
        smooth_levels = np.linspace(vmin, vmax, 1000)
        ax.contourf(Y_plot, X_plot, third_octave_oaspl_full, levels=smooth_levels, cmap=cmap)
        im = ax.pcolormesh(
            Y_plot, X_plot, third_octave_oaspl_full,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='gouraud',
            zorder=1,
        )
        
        # Add colorbar
        cbar = fig.colorbar(
            im, ax=ax, shrink=1, aspect=10, pad=0.05,
            ticks=np.arange(vmin, vmax + 10, 10),
        )
        cbar.set_label('OASPL [dB(A)]', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Add propeller disk circle
        angles = np.linspace(-np.pi, np.pi, 100)
        r_disk = 10 * self.propeller.geometry['tip_radius']
        ax.plot(r_disk * np.cos(angles), r_disk * np.sin(angles), '--', c='g', linewidth=1)
        
        # Add contour lines if requested
        if contour_levels:
            label_coords = [(2.0, 2.0), (2.0, 1.0), (2.0, 3.0), (2.0, 3.0), (2.0, 3.0), (2.0, 3.0)]
            
            for level, coord in zip(contour_levels, label_coords):
                cntr = ax.contour(
                    Y_plot, X_plot, third_octave_oaspl_full,
                    levels=[level], colors='white',
                    linewidths=0.8, alpha=0.5,
                )
                ax.clabel(cntr, inline=True, fontsize=12, fmt='%1.0f dB', manual=[coord])
        
        ax.set_xlabel('r [m]', fontsize=12)
        ax.set_ylabel('Z [m]', fontsize=12)
        ax.grid(True, alpha=0.15, color='white')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        
        plt.show()

    def plot_stress_distribution(
        self,
        sigma_c: np.ndarray,
        sigma_b: np.ndarray,
        stress_calculator: BladeStressCalculator,
        figsize: Tuple[float, float] = (4.5, 3),
        cmap: str = 'viridis',
        save_path: Optional[str] = None,
    ) -> None:
        '''Plot 3D blade stress distribution.
        
        Visualizes pre-calculated stress data on blade surface.
        
        Args:
            sigma_c: Centrifugal stress array (Pa).
            sigma_b: Bending stress array (Pa).
            stress_calculator: BladeStressCalculator instance for coordinate transformations.
            figsize: Figure size tuple (width, height) in inches.
            cmap: Colormap name (default 'viridis').
            save_path: Optional path to save figure as PNG.
        '''
        # Prepare geometry data
        geom = self.propeller.geometry
        r = np.asarray(geom['r'])
        chord = np.asarray(geom['chord'])
        twist = np.radians(np.asarray(geom['twist']))
        airfoils = geom['airfoil']
        
        sigma_c = np.asarray(sigma_c)
        sigma_b = np.asarray(sigma_b)
        
        # Expand sigma_c if needed to match sigma_b dimensions
        if sigma_b.ndim == 2 and sigma_c.ndim == 1:
            sigma_c = sigma_c[:, np.newaxis]
        
        if sigma_b.ndim == 1:
            sigma_total = sigma_c + sigma_b
            sigma_total = sigma_total[:, np.newaxis]
        else:
            sigma_total = sigma_b + sigma_c
        
        n_sections = len(r)
        n_points = airfoils[0].shape[0]
        
        X = np.zeros((n_sections, n_points))
        Y = np.zeros((n_sections, n_points))
        Z = np.zeros((n_sections, n_points))
        S = np.zeros((n_sections, n_points))
        
        # Build blade surface coordinates and stress values
        for i in range(n_sections):
            coords = np.asarray(airfoils[i])
            x_local = coords[:, 0] * chord[i]
            z_local = coords[:, 1] * chord[i]
            x_com, z_com = stress_calculator._compute_com(x_local, z_local)
            x_local = x_local - x_com
            z_local = z_local - z_com
            
            if hasattr(self.propeller, 'com_shift_forward') and hasattr(self.propeller, 'com_shift_up'):
                x_local = x_local + self.propeller.com_shift_forward[i] * chord[i]
                z_local = z_local + self.propeller.com_shift_up[i] * chord[i]
            
            cos_t = np.cos(twist[i])
            sin_t = np.sin(twist[i])
            x_rot = x_local * cos_t + z_local * sin_t
            z_rot = -x_local * sin_t + z_local * cos_t
            
            X[i, :] = x_rot
            Y[i, :] = r[i]
            Z[i, :] = z_rot
            
            if sigma_total.ndim == 2 and sigma_total.shape[1] == n_points:
                S[i, :] = sigma_total[i, :]
            else:
                S[i, :] = sigma_total[i]
        
        # Interpolate to finer grid
        r_fine = np.linspace(r.min(), r.max(), 50)
        X_fine = np.zeros((len(r_fine), n_points))
        Z_fine = np.zeros((len(r_fine), n_points))
        S_fine = np.zeros((len(r_fine), n_points))
        
        for j in range(n_points):
            X_fine[:, j] = np.interp(r_fine, r, X[:, j])
            Z_fine[:, j] = np.interp(r_fine, r, Z[:, j])
            S_fine[:, j] = np.interp(r_fine, r, S[:, j])
        
        Y_fine = np.repeat(r_fine[:, np.newaxis], n_points, axis=1)
        S_mpa = S_fine / 1e6
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        cmap_obj = mpl.colormaps[cmap]
        norm = mpl.colors.Normalize(vmin=np.nanmin(S_mpa), vmax=np.nanmax(S_mpa))
        ax.plot_surface(
            X_fine, Y_fine, Z_fine,
            facecolors=cmap_obj(norm(S_mpa)),
            rstride=1, cstride=1,
            linewidth=0, antialiased=True,
            shade=False, edgecolor='none',
        )
        
        ax.set_axis_off()
        ax.set_box_aspect((np.ptp(X_fine), np.ptp(Y_fine), np.ptp(Z_fine)))
        
        # Add colorbar
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(S_mpa)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.4, pad=-0.05, aspect=10)
        cbar.set_label('Stress [MPa]')
        
        ax.view_init(elev=15, azim=-30)
        plt.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
        
        plt.show()
