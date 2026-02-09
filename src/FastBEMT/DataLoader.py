from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Dict


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find repo root (pyproject.toml).")


def _repo_root() -> Path:
    return _find_repo_root(Path.cwd())


def load_propeller_dict(name: str) -> Dict[str, Any]:
    """Load a propeller dictionary from Datasets/Propellers."""
    path = _repo_root() / "Datasets" / "Propellers" / f"{name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def figures_dir() -> Path:
    """Return the Figures directory path inside the repo."""
    return _repo_root() / "Figures"
