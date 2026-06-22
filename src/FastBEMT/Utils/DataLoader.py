from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any

import numpy as np


def _find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking upward to ``pyproject.toml``."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find repo root (pyproject.toml).")


def _repo_root() -> Path:
    """Return the repository root for the current working directory."""
    return _find_repo_root(Path.cwd())


def load_propeller_dict(name: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Load one propeller pickle, or all pickles in a dataset directory."""
    path = _repo_root() / "Datasets" / name
    if path.is_dir():
        return [
            (pkl_file.stem, _load_pickle(pkl_file))
            for pkl_file in sorted(path.glob("*.pkl"))
        ]

    if not path.suffix:
        path = path.with_suffix(".pkl")
    return [(path.stem, _load_pickle(path))]


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return normalize_propeller_geometry(pickle.load(file))


def normalize_propeller_geometry(
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated new-convention propeller geometry dictionary."""
    required_keys = {
        "airfoils",
        "chord",
        "n_blades",
        "r",
        "rake",
        "sweep",
        "twist",
    }
    missing_keys = sorted(required_keys.difference(geometry))
    if missing_keys:
        raise ValueError(
            "Propeller geometry must use the new convention and contain "
            f"{', '.join(missing_keys)}."
        )

    r = _geometry_vector(geometry, "r")
    chord = _geometry_vector(geometry, "chord")
    rake = _geometry_vector(geometry, "rake")
    sweep = _geometry_vector(geometry, "sweep")
    twist = _geometry_vector(geometry, "twist")
    airfoils = np.asarray(geometry["airfoils"], dtype=np.float64)
    if airfoils.ndim != 3 or airfoils.shape[2] != 2:
        raise ValueError(
            "geometry['airfoils'] must have shape (sections, points, 2)."
        )

    section_count = r.shape[0]
    for name, values in (
        ("chord", chord),
        ("rake", rake),
        ("sweep", sweep),
        ("twist", twist),
    ):
        if values.shape != (section_count,):
            raise ValueError(
                f"geometry['{name}'] must contain {section_count} entries."
            )
    if airfoils.shape[0] != section_count:
        raise ValueError(
            "geometry['airfoils'] must contain one airfoil per radial section."
        )

    return {
        "r": r,
        "dr": _radial_widths(r),
        "chord": chord,
        "twist": twist,
        "airfoils": [
            np.array(airfoil, dtype=np.float64, copy=True)
            for airfoil in airfoils
        ],
        "sweep": sweep,
        "rake": rake,
        "n_blades": int(geometry["n_blades"]),
        "tip_radius": float(geometry.get("tip_radius", r[-1])),
        "hub_radius": float(geometry.get("hub_radius", r[0])),
    }


def _geometry_vector(
    geometry: Mapping[str, Any],
    name: str,
) -> np.ndarray:
    values = np.asarray(geometry[name], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"geometry['{name}'] must be one-dimensional.")
    if values.size == 0:
        raise ValueError(f"geometry['{name}'] must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"geometry['{name}'] must contain only finite values.")
    return values.copy()


def _radial_widths(r: np.ndarray) -> np.ndarray:
    if r.size < 2:
        raise ValueError("geometry['r'] must contain at least two sections.")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError("geometry['r'] must be strictly increasing.")

    radial_edges = np.concatenate(
        ([r[0]], 0.5 * (r[:-1] + r[1:]), [r[-1]])
    )
    return np.diff(radial_edges)


def figures_dir() -> Path:
    """Return the repository Figures directory."""
    return _repo_root() / "Figures"
