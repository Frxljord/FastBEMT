import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Tuple, Union
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
        blade_passing_freq = 1 / self.propeller.simulation.blade_passing_period

        ax_spl = fig.add_subplot(gs[:, 1])

        for idx in range(n_obs):
            ax_time = fig.add_subplot(gs[idx, 0])
            color = colors[idx % len(colors)]

            ax_time.plot(
                self.propeller.t[idx, :],
                self.propeller.p_m[idx, :],
                label="Monopole",
                linestyle="-",
                marker="o",
                markersize=1.5,
                alpha=0.4,
            )
            ax_time.plot(
                self.propeller.t[idx, :],
                self.propeller.p_d[idx, :],
                label="Dipole",
                linestyle="-",
                marker="s",
                markersize=1.5,
                alpha=0.4,
            )
            ax_time.plot(
                self.propeller.t[idx, :],
                self.propeller.p_tot[idx, :],
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
                self.propeller.freq[idx, 1:],
                self.propeller.spl[idx, 1:],
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

    @staticmethod
    def combine_level_maps(
        *levels: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Log-sum overall level maps in dB."""
        if not levels:
            raise ValueError("At least one level map is required.")

        total_power: np.ndarray | None = None
        for level in levels:
            if isinstance(level, torch.Tensor):
                level_array = level.detach().cpu().numpy()
            else:
                level_array = np.asarray(level)
            power = 10.0 ** (level_array.astype(float) / 10.0)
            total_power = power if total_power is None else total_power + power

        return 10.0 * np.log10(np.maximum(total_power, np.finfo(float).tiny))

    @staticmethod
    def _level_array(levels: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert level data to a NumPy array without changing values."""
        if isinstance(levels, torch.Tensor):
            return levels.detach().cpu().numpy()
        return np.asarray(levels)

    @staticmethod
    def _map_grid(
        levels: Union[np.ndarray, torch.Tensor],
        *,
        grid_size: int,
        mirror: bool,
    ) -> np.ndarray:
        """Reshape a flat map and apply the same mirroring as acoustic maps."""
        level_array = Plotter._level_array(levels)
        expected_size = grid_size * grid_size
        if level_array.size != expected_size:
            raise ValueError(
                f"levels must contain {expected_size} values; "
                f"got {level_array.size}."
            )

        level_grid = level_array.reshape(grid_size, grid_size)
        if mirror:
            return np.vstack([level_grid[::-1, :], level_grid[1:, :]])
        return level_grid

    @staticmethod
    def _contour_levels(
        contour_levels: Optional[Union[float, Iterable[float]]],
    ) -> Tuple[float, ...]:
        """Normalize contour input to zero or more finite levels."""
        if contour_levels is None:
            return ()
        if isinstance(contour_levels, (str, bytes)):
            raise TypeError(
                "contour_levels must be a number or an iterable of numbers."
            )

        try:
            levels = np.asarray(contour_levels, dtype=float)
        except (TypeError, ValueError) as exc:
            try:
                levels = np.asarray(tuple(contour_levels), dtype=float)
            except TypeError:
                raise TypeError(
                    "contour_levels must be a number or an iterable of numbers."
                ) from exc
            except ValueError as value_exc:
                raise TypeError(
                    "contour_levels must be a number or an iterable of numbers."
                ) from value_exc

        if levels.ndim == 0:
            normalized = (float(levels),)
        elif levels.ndim == 1:
            normalized = tuple(float(level) for level in levels)
        else:
            raise ValueError(
                "contour_levels must be a number or a one-dimensional "
                "iterable of numbers."
            )

        if not all(np.isfinite(normalized)):
            raise ValueError("contour_levels must contain only finite values.")
        return normalized

    def plot_acoustic_maps(
        self,
        levels_by_title: Mapping[str, Union[np.ndarray, torch.Tensor]],
        *,
        grid_size: int = 26,
        domain_size: float = 5.0,
        metric: str = "oaspl",
        columns: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = "magma",
        contour_levels: Optional[Union[float, Iterable[float]]] = None,
        mirror: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot one or more acoustic level maps with shared map styling."""
        metric = metric.lower()
        if metric not in {"ospl", "oaspl"}:
            raise ValueError("metric must be 'ospl' or 'oaspl'.")
        if not levels_by_title:
            raise ValueError("levels_by_title must contain at least one map.")

        titles = list(levels_by_title)
        maps = [
            self._map_grid(
                levels_by_title[title],
                grid_size=grid_size,
                mirror=mirror,
            )
            for title in titles
        ]
        contour_level_values = self._contour_levels(contour_levels)

        columns = max(1, min(int(columns), len(maps)))
        rows = int(np.ceil(len(maps) / columns))
        if figsize is None:
            figsize = (4.2 * columns, 3.2 * rows)
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        vmin = 0.0
        vmax = max(
            10.0,
            np.ceil(max(float(np.nanmax(values)) for values in maps) / 10.0)
            * 10.0,
        )

        x_range_plot = np.linspace(-domain_size, domain_size, grid_size)
        y_count = 2 * grid_size - 1 if mirror else grid_size
        y_range_plot = np.linspace(-domain_size, domain_size, y_count)
        x_plot, y_plot = np.meshgrid(x_range_plot, y_range_plot)
        smooth_levels = np.linspace(vmin, vmax, 1000)

        angles = np.linspace(-np.pi, np.pi, 100)
        disk_radius = 10.0 * self.propeller.geometry["tip_radius"]
        image = None
        label_coords = [
            (2.0, 2.0),
            (2.0, 1.0),
            (2.0, 3.0),
            (2.0, 3.0),
            (2.0, 3.0),
            (2.0, 3.0),
        ]

        for axis, title, level_map in zip(axes.flat, titles, maps):
            axis.contourf(
                y_plot,
                x_plot,
                level_map,
                levels=smooth_levels,
                cmap=cmap,
            )
            image = axis.pcolormesh(
                y_plot,
                x_plot,
                level_map,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="gouraud",
                zorder=1,
            )
            axis.plot(
                disk_radius * np.cos(angles),
                disk_radius * np.sin(angles),
                "--",
                c="g",
                linewidth=1,
            )

            if contour_level_values:
                for index, level in enumerate(contour_level_values):
                    coord = label_coords[index % len(label_coords)]
                    contour = axis.contour(
                        y_plot,
                        x_plot,
                        level_map,
                        levels=[level],
                        colors="white",
                        linewidths=0.8,
                        alpha=0.5,
                    )
                    axis.clabel(
                        contour,
                        inline=True,
                        fontsize=12,
                        fmt="%1.0f dB",
                        manual=[coord],
                    )

            axis.set_title(title, fontsize=12)
            axis.set_xlabel("r [m]", fontsize=12)
            axis.set_ylabel("Z [m]", fontsize=12)
            axis.tick_params(axis="both", labelsize=12)
            axis.grid(True, alpha=0.15, color="white")

        for axis in axes.flat[len(maps):]:
            axis.set_axis_off()

        colorbar = fig.colorbar(
            image,
            ax=axes,
            shrink=1,
            aspect=10,
            pad=0.05,
            ticks=np.arange(vmin, vmax + 10.0, 10.0),
        )
        colorbar_label = (
            "OASPL [dB(A)]" if metric == "oaspl" else "OSPL [dB]"
        )
        colorbar.set_label(colorbar_label, fontsize=12)
        colorbar.ax.tick_params(labelsize=12)

        if save_path is not None:
            plt.savefig(save_path, dpi=200)

        plt.show()

    def plot_acoustic_map(
        self,
        grid_size: int = 26,
        domain_size: float = 5.0,
        noise_type: str = 'total',
        metric: str = 'oaspl',
        levels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        figsize: Tuple[float, float] = (4.5, 2),
        cmap: str = 'magma',
        contour_levels: Optional[Union[float, Iterable[float]]] = None,
        mirror: Optional[bool] = True,
        save_path: Optional[str] = None,
    ) -> None:
        '''Plot 2D acoustic radiation pattern as contour map.
        
        Visualizes pre-calculated third-octave acoustic data stored on the
        propeller. These fields may come from the combined aeroacoustic
        pipeline or from a directly evaluated F1A object.
        
        Args:
            grid_size: Number of grid points in each direction (default 26).
            domain_size: Size of domain in meters (default 5.0).
            noise_type: Acoustic source to plot: ``"f1a"``, ``"bpm"``, or
                ``"total"``.
            metric: Overall level to plot: unweighted ``"ospl"`` or
                A-weighted ``"oaspl"``.
            levels: Optional precomputed overall levels with
                ``grid_size**2`` entries. When supplied, these values are
                plotted directly instead of reading a spectrum from the
                propeller.
            figsize: Figure size tuple (width, height) in inches.
            cmap: Colormap name (default 'magma').
            contour_levels: Optional contour level or iterable of contour
                levels in dB.
            save_path: Optional path to save figure as PDF.
        '''
        noise_type = noise_type.lower()
        metric = metric.lower()
        spectrum_attributes = {
            'f1a': 'third_octave_f1a_spl',
            'bpm': 'third_octave_bpm_spl',
            'total': 'third_octave_total_spl',
        }
        if noise_type not in spectrum_attributes:
            raise ValueError(
                "noise_type must be 'f1a', 'bpm', or 'total'."
            )
        if metric not in {'ospl', 'oaspl'}:
            raise ValueError("metric must be 'ospl' or 'oaspl'.")

        if levels is None:
            spectrum_name = spectrum_attributes[noise_type]
            if not hasattr(self.propeller, spectrum_name):
                raise RuntimeError(
                    f"Propeller has no {spectrum_name} data to plot."
                )
            spectrum = getattr(self.propeller, spectrum_name)
            if spectrum is None:
                raise RuntimeError(
                    f"Propeller has no {spectrum_name} data to plot."
                )

            from ..Aeroacoustics.Utils import third_octave_spectrum_to_overall_level

            overall_level = third_octave_spectrum_to_overall_level(
                spectrum,
                self.propeller.third_octave_freqs,
                weighted=metric == 'oaspl',
            )
            overall_level = overall_level.detach().cpu().numpy()
        elif isinstance(levels, torch.Tensor):
            overall_level = levels.detach().cpu().numpy()
        else:
            overall_level = np.asarray(levels)

        self.plot_acoustic_maps(
            {noise_type.upper(): overall_level},
            grid_size=grid_size,
            domain_size=domain_size,
            metric=metric,
            columns=1,
            figsize=figsize,
            cmap=cmap,
            contour_levels=contour_levels,
            mirror=bool(mirror),
            save_path=save_path,
        )

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
