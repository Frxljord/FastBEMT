from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from .Stress import BladeStressCalculator

if TYPE_CHECKING:
    from ..Propeller import Propeller


LevelArray = np.ndarray | torch.Tensor
FigureSize = tuple[float, float]


class Plotter:
    """Visualization utilities for propeller acoustic and stress results."""

    def __init__(self, propeller: Propeller) -> None:
        self.propeller = propeller

    def plot_observer_report(self) -> None:
        """Plot observer pressure histories and SPL spectra."""
        n_observers = self.propeller.observer_positions.shape[0]
        fig = plt.figure(figsize=(15, 3.5 * n_observers))
        grid = gridspec.GridSpec(n_observers, 2, figure=fig)

        fontsize = 10
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        blade_passing_freq = 1.0 / self.propeller.simulation.blade_passing_period
        frequencies = self._as_numpy(getattr(self.propeller, "freq"))
        oaspl = self._as_numpy(getattr(self.propeller, "oaspl"))
        spl = self._as_numpy(getattr(self.propeller, "spl"))
        ax_spl = fig.add_subplot(grid[:, 1])

        for observer_index in range(n_observers):
            color = colors[observer_index % len(colors)]
            ax_time = fig.add_subplot(grid[observer_index, 0])
            self._plot_pressure_history(ax_time, observer_index, color, fontsize)

            frequency_row = (
                frequencies
                if frequencies.ndim == 1
                else frequencies[observer_index]
            )
            label = (
                f"Observer {observer_index} - "
                f"OASPL {oaspl[observer_index]:.2f} dB(A)"
            )
            ax_spl.semilogx(
                frequency_row[1:],
                spl[observer_index, 1:],
                color=color,
                linestyle="-",
                marker=".",
                markersize=2,
                label=label,
                alpha=0.8,
            )

        ax_spl.set_title("SPL Spectrum", fontsize=fontsize, fontweight="bold")
        ax_spl.set_xlabel("Frequency [Hz]", fontsize=fontsize)
        ax_spl.set_ylabel("SPL [dB]", fontsize=fontsize)
        for index, harmonic in enumerate(
            np.arange(blade_passing_freq, np.max(frequencies), blade_passing_freq)
        ):
            ax_spl.axvline(
                harmonic,
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.2,
                label="Blade-passing harmonic" if index == 0 else None,
            )
        ax_spl.grid(True, which="both", alpha=0.3)
        ax_spl.legend(fontsize=fontsize, loc="lower left")

        plt.tight_layout()
        plt.show()

    def _plot_pressure_history(
        self,
        axis: plt.Axes,
        observer_index: int,
        color: str,
        fontsize: int,
    ) -> None:
        time = self._as_numpy(self.propeller.t)[observer_index, :]
        p_m = self._as_numpy(self.propeller.p_m)[observer_index, :]
        p_d = self._as_numpy(self.propeller.p_d)[observer_index, :]
        p_tot = self._as_numpy(self.propeller.p_tot)[observer_index, :]
        axis.plot(
            time,
            p_m,
            label="Monopole pressure",
            linestyle="-",
            marker="o",
            markersize=1.5,
            alpha=0.4,
        )
        axis.plot(
            time,
            p_d,
            label="Dipole pressure",
            linestyle="-",
            marker="s",
            markersize=1.5,
            alpha=0.4,
        )
        axis.plot(
            time,
            p_tot,
            color=color,
            linestyle="-",
            marker="d",
            markersize=2,
            linewidth=1.2,
            label="Total pressure",
        )
        axis.set_title(
            f"Observer {observer_index} Pressure",
            fontsize=fontsize,
            fontweight="bold",
        )
        axis.set_xlabel("Time [s]", fontsize=fontsize)
        axis.set_ylabel("Pressure [Pa]", fontsize=fontsize)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize="small")

    @staticmethod
    def combine_level_maps(*levels: LevelArray) -> np.ndarray:
        """Log-sum overall level maps in dB."""
        if not levels:
            raise ValueError("At least one level map is required.")

        total_power: np.ndarray | None = None
        for level in levels:
            power = 10.0 ** (Plotter._as_numpy(level).astype(float) / 10.0)
            total_power = power if total_power is None else total_power + power
        return 10.0 * np.log10(np.maximum(total_power, np.finfo(float).tiny))

    @staticmethod
    def _as_numpy(values: LevelArray) -> np.ndarray:
        """Convert arrays or tensors to NumPy without changing values."""
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    @staticmethod
    def _map_grid(levels: LevelArray, *, grid_size: int, mirror: bool) -> np.ndarray:
        """Reshape a flat map and apply the acoustic-map mirroring convention."""
        level_array = Plotter._as_numpy(levels)
        expected_size = grid_size * grid_size
        if level_array.size != expected_size:
            raise ValueError(
                f"levels must contain {expected_size} values; got {level_array.size}."
            )

        level_grid = level_array.reshape(grid_size, grid_size)
        if mirror:
            return np.vstack([level_grid[::-1, :], level_grid[1:, :]])
        return level_grid

    @staticmethod
    def _contour_levels(
        contour_levels: float | Iterable[float] | None,
    ) -> tuple[float, ...]:
        """Normalize contour input to zero or more finite dB levels."""
        if contour_levels is None:
            return ()
        if isinstance(contour_levels, (str, bytes)):
            raise TypeError("contour_levels must be numeric.")

        try:
            levels = np.asarray(contour_levels, dtype=float)
        except (TypeError, ValueError):
            levels = np.asarray(tuple(contour_levels), dtype=float)

        if levels.ndim == 0:
            normalized = (float(levels),)
        elif levels.ndim == 1:
            normalized = tuple(float(level) for level in levels)
        else:
            raise ValueError("contour_levels must be scalar or one-dimensional.")
        if not all(np.isfinite(normalized)):
            raise ValueError("contour_levels must contain only finite values.")
        return normalized

    def plot_acoustic_maps(
        self,
        levels_by_title: Mapping[str, LevelArray],
        *,
        grid_size: int = 26,
        domain_size: float = 5.0,
        metric: str = "oaspl",
        columns: int = 3,
        figsize: FigureSize | None = None,
        cmap: str = "magma",
        contour_levels: float | Iterable[float] | None = None,
        mirror: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot one or more acoustic level maps with shared styling."""
        metric = metric.lower()
        if metric not in {"ospl", "oaspl"}:
            raise ValueError("metric must be 'ospl' or 'oaspl'.")
        if not levels_by_title:
            raise ValueError("levels_by_title must contain at least one map.")

        titles = list(levels_by_title)
        maps = [
            self._map_grid(levels_by_title[title], grid_size=grid_size, mirror=mirror)
            for title in titles
        ]
        contour_level_values = self._contour_levels(contour_levels)
        columns = max(1, min(int(columns), len(maps)))
        rows = int(np.ceil(len(maps) / columns))
        figsize = figsize or (4.2 * columns, 3.2 * rows)

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )
        image = self._draw_level_maps(
            axes,
            titles,
            maps,
            grid_size=grid_size,
            domain_size=domain_size,
            cmap=cmap,
            contour_levels=contour_level_values,
            mirror=mirror,
        )
        colorbar = fig.colorbar(
            image,
            ax=axes,
            shrink=1,
            aspect=10,
            pad=0.05,
            ticks=np.arange(0.0, image.norm.vmax + 10.0, 10.0),
        )
        colorbar.set_label(
            "OASPL [dB(A)]" if metric == "oaspl" else "OSPL [dB]",
            fontsize=12,
        )
        colorbar.ax.tick_params(labelsize=12)

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.show()

    def _draw_level_maps(
        self,
        axes: np.ndarray,
        titles: list[str],
        maps: list[np.ndarray],
        *,
        grid_size: int,
        domain_size: float,
        cmap: str,
        contour_levels: tuple[float, ...],
        mirror: bool,
    ) -> mpl.collections.QuadMesh:
        vmax = max(
            10.0,
            np.ceil(max(float(np.nanmax(values)) for values in maps) / 10.0) * 10.0,
        )
        x_range = np.linspace(-domain_size, domain_size, grid_size)
        y_count = 2 * grid_size - 1 if mirror else grid_size
        y_range = np.linspace(-domain_size, domain_size, y_count)
        x_plot, y_plot = np.meshgrid(x_range, y_range)
        smooth_levels = np.linspace(0.0, vmax, 1000)
        disk_angles = np.linspace(-np.pi, np.pi, 100)
        disk_radius = 10.0 * self.propeller.geometry["tip_radius"]
        image = None

        for axis, title, level_map in zip(axes.flat, titles, maps):
            axis.contourf(y_plot, x_plot, level_map, levels=smooth_levels, cmap=cmap)
            image = axis.pcolormesh(
                y_plot,
                x_plot,
                level_map,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                shading="gouraud",
                zorder=1,
            )
            axis.plot(
                disk_radius * np.cos(disk_angles),
                disk_radius * np.sin(disk_angles),
                "--",
                c="g",
                linewidth=1,
            )
            self._draw_contours(axis, y_plot, x_plot, level_map, contour_levels)
            axis.set_title(title, fontsize=12)
            axis.set_xlabel("r [m]", fontsize=12)
            axis.set_ylabel("Z [m]", fontsize=12)
            axis.tick_params(axis="both", labelsize=12)
            axis.grid(True, alpha=0.15, color="white")

        for axis in axes.flat[len(maps):]:
            axis.set_axis_off()
        if image is None:
            raise RuntimeError("No acoustic maps were drawn.")
        return image

    @staticmethod
    def _draw_contours(
        axis: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        contour_levels: tuple[float, ...],
    ) -> None:
        label_coords = ((2.0, 2.0), (2.0, 1.0), (2.0, 3.0))
        for index, level in enumerate(contour_levels):
            contour = axis.contour(
                x,
                y,
                values,
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
                manual=[label_coords[index % len(label_coords)]],
            )

    def plot_acoustic_map(
        self,
        grid_size: int = 26,
        domain_size: float = 5.0,
        noise_type: str = "total",
        metric: str = "oaspl",
        levels: LevelArray | None = None,
        figsize: FigureSize = (4.5, 2),
        cmap: str = "magma",
        contour_levels: float | Iterable[float] | None = None,
        mirror: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Plot one 2D acoustic radiation map."""
        noise_type = noise_type.lower()
        metric = metric.lower()
        if noise_type not in {"f1a", "bpm", "total"}:
            raise ValueError("noise_type must be 'f1a', 'bpm', or 'total'.")
        if metric not in {"ospl", "oaspl"}:
            raise ValueError("metric must be 'ospl' or 'oaspl'.")

        overall_level = (
            self._overall_level_from_propeller(noise_type, metric)
            if levels is None
            else self._as_numpy(levels)
        )
        self.plot_acoustic_maps(
            {noise_type.upper(): overall_level},
            grid_size=grid_size,
            domain_size=domain_size,
            metric=metric,
            columns=1,
            figsize=figsize,
            cmap=cmap,
            contour_levels=contour_levels,
            mirror=mirror,
            save_path=save_path,
        )

    def _overall_level_from_propeller(self, noise_type: str, metric: str) -> np.ndarray:
        spectrum_attributes = {
            "f1a": "third_octave_f1a_spl",
            "bpm": "third_octave_bpm_spl",
            "total": "third_octave_total_spl",
        }
        spectrum_name = spectrum_attributes[noise_type]
        spectrum = getattr(self.propeller, spectrum_name, None)
        if spectrum is None:
            raise RuntimeError(f"Propeller has no {spectrum_name} data to plot.")

        from ..Aeroacoustics.Utils import third_octave_spectrum_to_overall_level

        overall_level = third_octave_spectrum_to_overall_level(
            spectrum,
            self.propeller.third_octave_freqs,
            weighted=metric == "oaspl",
        )
        return overall_level.detach().cpu().numpy()

    def plot_stress_distribution(
        self,
        sigma_c: np.ndarray,
        sigma_b: np.ndarray,
        stress_calculator: BladeStressCalculator,
        figsize: FigureSize = (4.5, 3),
        cmap: str = "viridis",
        save_path: str | None = None,
    ) -> None:
        """Plot a 3D blade stress distribution."""
        x_fine, y_fine, z_fine, stress_mpa = self._stress_surface(
            sigma_c,
            sigma_b,
            stress_calculator,
        )
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(111, projection="3d")

        cmap_obj = mpl.colormaps[cmap]
        norm = mpl.colors.Normalize(
            vmin=np.nanmin(stress_mpa),
            vmax=np.nanmax(stress_mpa),
        )
        axis.plot_surface(
            x_fine,
            y_fine,
            z_fine,
            facecolors=cmap_obj(norm(stress_mpa)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
            edgecolor="none",
        )
        axis.set_axis_off()
        axis.set_box_aspect((np.ptp(x_fine), np.ptp(y_fine), np.ptp(z_fine)))

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(stress_mpa)
        colorbar = fig.colorbar(mappable, ax=axis, shrink=0.4, pad=-0.05, aspect=10)
        colorbar.set_label("Stress [MPa]")

        axis.view_init(elev=15, azim=-30)
        plt.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.show()

    def _stress_surface(
        self,
        sigma_c: np.ndarray,
        sigma_b: np.ndarray,
        stress_calculator: BladeStressCalculator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        geom = self.propeller.geometry
        radius = np.asarray(geom["r"])
        chord = np.asarray(geom["chord"])
        twist = np.radians(np.asarray(geom["twist"]))
        airfoils = geom["airfoil"]
        sigma_total = self._combined_stress(sigma_c, sigma_b)

        n_sections = len(radius)
        n_points = airfoils[0].shape[0]
        x_sections = np.zeros((n_sections, n_points))
        z_sections = np.zeros((n_sections, n_points))
        stress_sections = np.zeros((n_sections, n_points))

        for section_index in range(n_sections):
            x_rot, z_rot = self._section_surface_coordinates(
                section_index,
                chord,
                twist,
                airfoils,
                stress_calculator,
            )
            x_sections[section_index, :] = x_rot
            z_sections[section_index, :] = z_rot
            if sigma_total.ndim == 2 and sigma_total.shape[1] == n_points:
                stress_sections[section_index, :] = sigma_total[section_index, :]
            else:
                stress_sections[section_index, :] = sigma_total[section_index]

        radius_fine = np.linspace(radius.min(), radius.max(), 50)
        x_fine = np.vstack(
            [
                np.interp(radius_fine, radius, x_sections[:, idx])
                for idx in range(n_points)
            ]
        ).T
        z_fine = np.vstack(
            [
                np.interp(radius_fine, radius, z_sections[:, idx])
                for idx in range(n_points)
            ]
        ).T
        stress_fine = np.vstack(
            [
                np.interp(radius_fine, radius, stress_sections[:, idx])
                for idx in range(n_points)
            ]
        ).T
        y_fine = np.repeat(radius_fine[:, np.newaxis], n_points, axis=1)
        return x_fine, y_fine, z_fine, stress_fine / 1e6

    @staticmethod
    def _combined_stress(sigma_c: np.ndarray, sigma_b: np.ndarray) -> np.ndarray:
        sigma_c = np.asarray(sigma_c)
        sigma_b = np.asarray(sigma_b)
        if sigma_b.ndim == 2 and sigma_c.ndim == 1:
            sigma_c = sigma_c[:, np.newaxis]
        if sigma_b.ndim == 1:
            sigma_b = sigma_b[:, np.newaxis]
        return sigma_b + sigma_c

    def _section_surface_coordinates(
        self,
        section_index: int,
        chord: np.ndarray,
        twist: np.ndarray,
        airfoils: list[np.ndarray],
        stress_calculator: BladeStressCalculator,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(airfoils[section_index])
        x_local = coords[:, 0] * chord[section_index]
        z_local = coords[:, 1] * chord[section_index]
        x_com, z_com = stress_calculator._compute_com(x_local, z_local)
        x_local = x_local - x_com
        z_local = z_local - z_com

        if hasattr(self.propeller, "com_shift_forward"):
            x_local = (
                x_local
                + self.propeller.com_shift_forward[section_index] * chord[section_index]
            )
        if hasattr(self.propeller, "com_shift_up"):
            z_local = (
                z_local
                + self.propeller.com_shift_up[section_index] * chord[section_index]
            )

        cos_twist = np.cos(twist[section_index])
        sin_twist = np.sin(twist[section_index])
        return (
            x_local * cos_twist + z_local * sin_twist,
            -x_local * sin_twist + z_local * cos_twist,
        )
