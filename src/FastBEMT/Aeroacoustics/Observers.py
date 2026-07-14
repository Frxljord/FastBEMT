from __future__ import annotations

import numpy as np


def uniform_observer_grid(size: float, nx: int, ny: int) -> np.ndarray:
    """Return an ``(nx * ny, 3)`` observer grid in the x-y plane."""
    x_range = np.linspace(-size, size, nx)
    y_range = np.linspace(0.0, size, ny)
    x, y = np.meshgrid(x_range, y_range)
    return np.stack([x.ravel(), y.ravel(), np.zeros_like(x).ravel()], axis=1)


def semicircular_observer_array(radius: float, n_points: int) -> np.ndarray:
    """Return a semicircular observer array in the x-y plane."""
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_points, endpoint=True)
    return np.stack(
        [
            radius * np.sin(angles),
            radius * np.cos(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    )
