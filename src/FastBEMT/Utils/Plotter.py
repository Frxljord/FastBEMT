from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from ..Propeller import Propeller


LevelArray = np.ndarray | torch.Tensor
FigureSize = tuple[float, float]
SavePath = str | PathLike[str]
ACOUSTIC_MAP_MIN_DB = -80.0


class Plotter:
    """Visualization utilities for propeller acoustic and stress results."""

    def __init__(self, propeller: Propeller) -> None:
        self.propeller = propeller

    @staticmethod
    def combine_level_maps(*levels: LevelArray) -> np.ndarray:
        """Combine decibel maps by summing their linear acoustic power."""
        total_power = sum(
            10.0 ** (Plotter._as_numpy(level).astype(float) / 10.0)
            for level in levels
        )
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
        level_grid = Plotter._as_numpy(levels).reshape(grid_size, grid_size)
        if mirror:
            return np.vstack([level_grid[::-1, :], level_grid[1:, :]])
        return level_grid

    @staticmethod
    def _contour_levels(
        contour_levels: float | Iterable[float] | None,
    ) -> tuple[float, ...]:
        """Normalize contour input to zero or more dB levels."""
        if contour_levels is None:
            return ()
        return tuple(np.atleast_1d(np.asarray(contour_levels, dtype=float)))

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
        shared_color_scale: bool = False,
        save_path: SavePath | None = None,
    ) -> None:
        """Plot one or more acoustic level maps.

        Args:
            levels_by_title: Mapping from subplot titles to flat level maps in dB.
            grid_size: Number of samples along each input-map dimension.
            domain_size: Positive and negative extent of each plot axis.
            metric: ``"oaspl"`` for a dB(A) colorbar, otherwise dB.
            columns: Maximum number of subplot columns.
            figsize: Figure width and height in inches.
            cmap: Matplotlib color-map name.
            contour_levels: Optional labeled contour levels in dB.
            mirror: Mirror the half-plane map across its first row.
            shared_color_scale: Use one color scale and colorbar for all maps.
            save_path: Optional image output path.
        """
        metric = metric.lower()
        titles = list(levels_by_title)
        maps = [
            self._map_grid(levels_by_title[title], grid_size=grid_size, mirror=mirror)
            for title in titles
        ]
        contour_level_values = self._contour_levels(contour_levels)
        columns = min(int(columns), len(maps))
        rows = int(np.ceil(len(maps) / columns))
        figsize = figsize or (4.2 * columns, 3.2 * rows)

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )
        images = self._draw_level_maps(
            axes,
            titles,
            maps,
            grid_size=grid_size,
            domain_size=domain_size,
            cmap=cmap,
            contour_levels=contour_level_values,
            mirror=mirror,
            shared_color_scale=shared_color_scale,
        )
        colorbar_label = "OASPL [dB(A)]" if metric == "oaspl" else "OSPL [dB]"
        active_axes = list(axes.flat[: len(maps)])
        colorbar_targets = (
            [(images[0], active_axes)]
            if shared_color_scale
            else list(zip(images, active_axes))
        )
        for image, target_axes in colorbar_targets:
            colorbar = fig.colorbar(
                image,
                ax=target_axes,
                shrink=1,
                aspect=10,
                pad=0.05,
                ticks=np.arange(image.norm.vmin, image.norm.vmax + 10.0, 10.0),
            )
            colorbar.set_label(colorbar_label, fontsize=12)
            colorbar.ax.tick_params(labelsize=12)

        if save_path is not None:
            fig.savefig(save_path, dpi=200)
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
        shared_color_scale: bool,
    ) -> list[mpl.collections.QuadMesh]:
        bounds = (
            [self._level_map_bounds(maps)] * len(maps)
            if shared_color_scale
            else [self._level_map_bounds([level_map]) for level_map in maps]
        )
        x_range = np.linspace(-domain_size, domain_size, grid_size)
        y_count = 2 * grid_size - 1 if mirror else grid_size
        y_range = np.linspace(-domain_size, domain_size, y_count)
        x_plot, y_plot = np.meshgrid(x_range, y_range)
        disk_angles = np.linspace(-np.pi, np.pi, 100)
        disk_radius = 10.0 * self.propeller.geometry["tip_radius"]
        images = []

        for axis, title, level_map, (vmin, vmax) in zip(
            axes.flat, titles, maps, bounds
        ):
            display_map = np.maximum(level_map, vmin)
            image = axis.pcolormesh(
                y_plot,
                x_plot,
                display_map,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="gouraud",
                zorder=1,
            )
            images.append(image)
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
        return images

    @staticmethod
    def _level_map_bounds(maps: list[np.ndarray]) -> tuple[float, float]:
        """Return rounded dB bounds for one color scale."""
        values = np.concatenate([level_map.ravel() for level_map in maps])
        vmax = max(10.0, np.ceil(float(np.nanmax(values)) / 10.0) * 10.0)
        vmin = max(
            ACOUSTIC_MAP_MIN_DB,
            np.floor(float(np.nanmin(values)) / 10.0) * 10.0,
        )
        if vmax <= vmin:
            vmax = vmin + 10.0
        return vmin, vmax

    @staticmethod
    def _draw_contours(
        axis: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        contour_levels: tuple[float, ...],
    ) -> None:
        for level in contour_levels:
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
            )

    def plot_acoustic_map(
        self,
        levels: LevelArray,
        *,
        title: str = "Acoustic level",
        grid_size: int = 26,
        domain_size: float = 5.0,
        metric: str = "oaspl",
        figsize: FigureSize = (4.5, 2),
        cmap: str = "magma",
        contour_levels: float | Iterable[float] | None = None,
        mirror: bool = True,
        save_path: SavePath | None = None,
    ) -> None:
        """Plot one acoustic level map supplied explicitly in dB.

        Args:
            levels: Flat acoustic level map.
            title: Subplot title.
            grid_size: Number of samples along each input-map dimension.
            domain_size: Positive and negative extent of each plot axis.
            metric: ``"oaspl"`` for a dB(A) colorbar, otherwise dB.
            figsize: Figure width and height in inches.
            cmap: Matplotlib color-map name.
            contour_levels: Optional labeled contour levels in dB.
            mirror: Mirror the half-plane map across its first row.
            save_path: Optional image output path.
        """
        self.plot_acoustic_maps(
            {title: levels},
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

    def plot_stress_distribution(
        self,
        centrifugal_stress: np.ndarray,
        bending_stress: np.ndarray,
        figsize: FigureSize = (4.5, 3),
        cmap: str = "viridis",
        save_path: SavePath | None = None,
    ) -> None:
        """Plot the combined stress over one blade surface.

        Args:
            centrifugal_stress: Spanwise centrifugal stress in Pa.
            bending_stress: Bending stress at every airfoil point in Pa.
            figsize: Figure width and height in inches.
            cmap: Matplotlib color-map name.
            save_path: Optional image output path.
        """
        x_fine, y_fine, z_fine, stress_mpa = self._stress_surface(
            centrifugal_stress,
            bending_stress,
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
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.show()

    def _stress_surface(
        self,
        centrifugal_stress: np.ndarray,
        bending_stress: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        geom = self.propeller.geometry
        radius = np.asarray(geom["r"])
        chord = np.asarray(geom["chord"])
        twist = np.radians(np.asarray(geom["twist"]))
        airfoils = geom["airfoils"]
        combined_stress = self._combined_stress(
            centrifugal_stress,
            bending_stress,
        )

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
            )
            x_sections[section_index, :] = x_rot
            z_sections[section_index, :] = z_rot
            stress_sections[section_index, :] = combined_stress[section_index, :]

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
    def _combined_stress(
        centrifugal_stress: np.ndarray,
        bending_stress: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(bending_stress) + np.asarray(centrifugal_stress)[:, None]

    def _section_surface_coordinates(
        self,
        section_index: int,
        chord: np.ndarray,
        twist: np.ndarray,
        airfoils: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(airfoils[section_index])
        x_local = coords[:, 0] * chord[section_index]
        z_local = coords[:, 1] * chord[section_index]
        x_com, z_com = self._polygon_centroid(x_local, z_local)
        x_local = x_local - x_com
        z_local = z_local - z_com

        geometry = self.propeller.geometry
        x_local = x_local - geometry["sweep"][section_index]
        z_local = z_local + geometry["rake"][section_index]

        cos_twist = np.cos(twist[section_index])
        sin_twist = np.sin(twist[section_index])
        return (
            x_local * cos_twist + z_local * sin_twist,
            -x_local * sin_twist + z_local * cos_twist,
        )

    @staticmethod
    def _polygon_centroid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return the centroid of a polygon."""
        x_current = x[:-1]
        y_current = y[:-1]
        x_next = x[1:]
        y_next = y[1:]
        cross = x_current * y_next - x_next * y_current
        scale = 3.0 * np.sum(cross)
        return (
            float(np.sum((x_current + x_next) * cross) / scale),
            float(np.sum((y_current + y_next) * cross) / scale),
        )
