"""Animate the global, blade, and airfoil coordinate frames in 3D."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from FastBEMT import Environment, Kinematics, Propeller, Simulation


AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
AXIS_NAMES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Animate the global, blade, and airfoil frames for one propeller "
            "blade section."
        )
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=REPO_ROOT / "Data" / "10x7E.pkl",
        help="Path to a pickled propeller geometry dictionary.",
    )
    parser.add_argument("--rpm", type=float, default=3000.0)
    parser.add_argument(
        "--section",
        type=int,
        default=-1,
        help="Radial section index to track. Negative indices are supported.",
    )
    parser.add_argument("--blade", type=int, default=0)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=120,
        help="Animation frames per revolution.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=40.0,
        help="Delay between animation frames in milliseconds.",
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.22,
        help="Frame-axis length as a fraction of propeller radius.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output path, normally ending in .gif or .mp4.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive Matplotlib window.",
    )
    return parser.parse_args()


def load_geometry(path: Path) -> dict:
    """Load a propeller geometry dictionary."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Geometry file does not exist: {path}")
    with path.open("rb") as geometry_file:
        geometry = pickle.load(geometry_file)
    if not isinstance(geometry, dict):
        raise TypeError("The geometry pickle must contain a dictionary.")
    return geometry


def normalize_index(index: int, size: int, name: str) -> int:
    """Resolve a possibly negative index and validate its range."""
    resolved = index + size if index < 0 else index
    if resolved < 0 or resolved >= size:
        raise IndexError(f"{name} index {index} is invalid for size {size}.")
    return resolved


def set_line_3d(
    line: Line2D,
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    """Move a Matplotlib 3D line segment."""
    line.set_data_3d(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
    )


def create_triad(
    ax,
    origin: np.ndarray,
    basis: np.ndarray,
    length: float,
    frame_suffix: str,
    linestyle: str,
    linewidth: float,
) -> tuple[list[Line2D], list]:
    """Draw a coordinate triad whose basis vectors are columns of ``basis``."""
    lines: list[Line2D] = []
    labels = []
    for axis_index, (axis_name, color) in enumerate(
        zip(AXIS_NAMES, AXIS_COLORS)
    ):
        endpoint = origin + length * basis[:, axis_index]
        line, = ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markevery=[1],
            markersize=3.5,
        )
        label = ax.text(
            endpoint[0],
            endpoint[1],
            endpoint[2],
            f"{axis_name}_{frame_suffix}",
            color=color,
            fontsize=9,
        )
        lines.append(line)
        labels.append(label)
    return lines, labels


def update_triad(
    lines: Sequence[Line2D],
    labels: Sequence,
    origin: np.ndarray,
    basis: np.ndarray,
    length: float,
) -> None:
    """Update a coordinate triad in place."""
    for axis_index, (line, label) in enumerate(zip(lines, labels)):
        endpoint = origin + length * basis[:, axis_index]
        set_line_3d(line, origin, endpoint)
        label.set_position_3d(endpoint)


def build_animation(
    kinematics: Kinematics,
    section_index: int,
    blade_index: int,
    interval: float,
    axis_scale: float,
) -> tuple[plt.Figure, FuncAnimation]:
    """Build the frame animation."""
    positions = (
        kinematics.section_position_global_frame.detach().cpu().numpy()
    )
    global_to_blade = (
        kinematics.global_to_blade_rotation_matrix.detach().cpu().numpy()
    )
    blade_to_airfoil = (
        kinematics.blade_to_airfoil_rotation_matrix.detach().cpu().numpy()
    )
    times = kinematics.source_times.detach().cpu().numpy()
    angles = kinematics.blade_angles.detach().cpu().numpy()

    radius = float(kinematics.propeller.geometry["tip_radius"])
    hub_radius = float(kinematics.propeller.geometry["hub_radius"])
    if interval <= 0.0:
        raise ValueError("interval must be greater than zero.")
    if axis_scale <= 0.0:
        raise ValueError("axis_scale must be greater than zero.")
    axis_length = axis_scale * radius
    airfoil_axis_length = 0.85 * axis_length
    plot_limit = 1.25 * max(radius, np.abs(positions).max())

    figure = plt.figure(figsize=(9, 8))
    ax = figure.add_subplot(111, projection="3d")
    ax.set_xlabel("Global x")
    ax.set_ylabel("Global y")
    ax.set_zlabel("Global z")
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_zlim(-plot_limit, plot_limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=24.0, azim=38.0)

    origin = np.zeros(3)
    create_triad(
        ax,
        origin,
        np.eye(3),
        axis_length,
        frame_suffix="g",
        linestyle="-",
        linewidth=2.0,
    )

    blade_basis = global_to_blade[0, blade_index].T
    airfoil_basis = blade_basis @ blade_to_airfoil[section_index].T
    section_origin = positions[0, blade_index, section_index]

    blade_lines, blade_labels = create_triad(
        ax,
        origin,
        blade_basis,
        axis_length,
        frame_suffix="b",
        linestyle="--",
        linewidth=2.0,
    )
    airfoil_lines, airfoil_labels = create_triad(
        ax,
        section_origin,
        airfoil_basis,
        airfoil_axis_length,
        frame_suffix="a",
        linestyle=":",
        linewidth=2.5,
    )

    blade_lines_all = []
    for blade in range(kinematics.nb):
        blade_positions = np.vstack((origin, positions[0, blade, :]))
        line, = ax.plot(
            blade_positions[:, 0],
            blade_positions[:, 1],
            blade_positions[:, 2],
            color="0.55" if blade != blade_index else "black",
            linewidth=1.2 if blade != blade_index else 2.4,
            alpha=0.7,
        )
        blade_lines_all.append(line)

    selected_point, = ax.plot(
        [section_origin[0]],
        [section_origin[1]],
        [section_origin[2]],
        marker="o",
        color="black",
        markersize=6,
    )
    section_trajectory = positions[:, blade_index, section_index]
    section_trajectory = np.vstack(
        (section_trajectory, section_trajectory[0])
    )
    trajectory, = ax.plot(
        section_trajectory[:, 0],
        section_trajectory[:, 1],
        section_trajectory[:, 2],
        color="0.35",
        linewidth=1.0,
        alpha=0.35,
    )

    frame_legend = [
        Line2D([0], [0], color="black", linestyle="-", label="Global frame"),
        Line2D([0], [0], color="black", linestyle="--", label="Blade frame"),
        Line2D([0], [0], color="black", linestyle=":", label="Airfoil frame"),
    ]
    axis_legend = [
        Line2D([0], [0], color=color, label=f"{name} axis")
        for name, color in zip(AXIS_NAMES, AXIS_COLORS)
    ]
    ax.legend(handles=frame_legend + axis_legend, loc="upper left")

    hub_circle_angle = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.plot(
        np.zeros_like(hub_circle_angle),
        hub_radius * np.cos(hub_circle_angle),
        hub_radius * np.sin(hub_circle_angle),
        color="0.5",
        linewidth=1.0,
        alpha=0.5,
    )

    def update(frame_index: int):
        current_blade_basis = global_to_blade[frame_index, blade_index].T
        current_airfoil_basis = (
            current_blade_basis @ blade_to_airfoil[section_index].T
        )
        current_section_origin = positions[
            frame_index, blade_index, section_index
        ]

        update_triad(
            blade_lines,
            blade_labels,
            origin,
            current_blade_basis,
            axis_length,
        )
        update_triad(
            airfoil_lines,
            airfoil_labels,
            current_section_origin,
            current_airfoil_basis,
            airfoil_axis_length,
        )

        for blade, line in enumerate(blade_lines_all):
            blade_positions = np.vstack(
                (origin, positions[frame_index, blade, :])
            )
            line.set_data_3d(
                blade_positions[:, 0],
                blade_positions[:, 1],
                blade_positions[:, 2],
            )

        selected_point.set_data_3d(
            [current_section_origin[0]],
            [current_section_origin[1]],
            [current_section_origin[2]],
        )
        angle_deg = np.rad2deg(angles[frame_index, blade_index]) % 360.0
        ax.set_title(
            "Propeller coordinate frames\n"
            f"blade={blade_index}, section={section_index}, "
            f"time={times[frame_index]:.5f} s, angle={angle_deg:.1f} deg"
        )
        return (
            *blade_lines,
            *airfoil_lines,
            *blade_lines_all,
            selected_point,
            trajectory,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=kinematics.nt,
        interval=interval,
        repeat=True,
        blit=False,
    )

    animation_paused = False

    def toggle_pause(event) -> None:
        nonlocal animation_paused
        if event.key != " ":
            return
        if animation_paused:
            animation.resume()
        else:
            animation.pause()
        animation_paused = not animation_paused

    figure.canvas.mpl_connect("key_press_event", toggle_pause)
    figure.text(
        0.5,
        0.015,
        "Press Space to pause/resume",
        ha="center",
        fontsize=9,
    )
    update(0)
    figure.tight_layout()
    return figure, animation


def main() -> None:
    """Load geometry and display or save the coordinate-frame animation."""
    args = parse_args()
    geometry = load_geometry(args.geometry)
    simulation = Simulation(
        revolutions=1,
        timesteps_per_revolution=args.timesteps,
        device="cpu",
    )
    propeller = Propeller(
        geometry=geometry,
        environment=Environment(),
        simulation=simulation,
    )
    kinematics = Kinematics(propeller, rpm=args.rpm)

    section_index = normalize_index(args.section, kinematics.ns, "section")
    blade_index = normalize_index(args.blade, kinematics.nb, "blade")
    figure, animation = build_animation(
        kinematics=kinematics,
        section_index=section_index,
        blade_index=blade_index,
        interval=args.interval,
        axis_scale=args.axis_scale,
    )

    if args.save is not None:
        output_path = args.save.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = 1000.0 / args.interval
        if output_path.suffix.lower() == ".gif":
            animation.save(output_path, writer=PillowWriter(fps=fps))
        else:
            animation.save(output_path, fps=fps)
        print(f"Saved animation to {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
