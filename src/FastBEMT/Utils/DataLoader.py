"""Load current-schema propeller geometry files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_propeller_geometry(path: str | Path) -> dict[str, Any]:
    """Load one current-schema propeller geometry pickle."""
    return _load_pickle(_resolve_repo_path(path, add_pickle_suffix=True))


def load_propeller_geometries(
    directory: str | Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Load all propeller geometry pickles in a directory."""
    path = _resolve_repo_path(directory)
    return [
        (pickle_path.stem, _load_pickle(pickle_path))
        for pickle_path in sorted(path.glob("*.pkl"))
    ]


def _resolve_repo_path(
    path: str | Path,
    *,
    add_pickle_suffix: bool = False,
) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = REPOSITORY_ROOT / resolved_path
    if add_pickle_suffix and not resolved_path.suffix:
        resolved_path = resolved_path.with_suffix(".pkl")
    return resolved_path


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return pickle.load(file)


def normalize_propeller_geometry(
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a current-schema geometry mapping to canonical NumPy arrays."""
    vectors = {
        name: np.asarray(geometry[name], dtype=np.float64).copy()
        for name in ("r", "chord", "twist", "sweep", "rake")
    }
    airfoils = np.asarray(geometry["airfoils"], dtype=np.float64)

    return {
        "r": vectors["r"],
        "dr": _radial_widths(vectors["r"]),
        "chord": vectors["chord"],
        "twist": vectors["twist"],
        "airfoils": [airfoil.copy() for airfoil in airfoils],
        "sweep": vectors["sweep"],
        "rake": vectors["rake"],
        "n_blades": int(geometry["n_blades"]),
        "tip_radius": float(geometry["tip_radius"]),
        "hub_radius": float(geometry["hub_radius"]),
    }


def _radial_widths(radial_positions: np.ndarray) -> np.ndarray:
    radial_edges = np.concatenate(
        (
            [radial_positions[0]],
            0.5 * (radial_positions[:-1] + radial_positions[1:]),
            [radial_positions[-1]],
        )
    )
    return np.diff(radial_edges)
