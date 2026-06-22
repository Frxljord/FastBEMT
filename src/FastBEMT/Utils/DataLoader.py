from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


def _find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking upward to ``pyproject.toml``."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find repo root (pyproject.toml).")


def _repo_root() -> Path:
    """Return the repository root for the current working directory."""
    return _find_repo_root(Path.cwd())


def load_propeller_dict(name: str) -> list[tuple[str, dict[str, Any]]]:
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
        return pickle.load(file)


def figures_dir() -> Path:
    """Return the repository Figures directory."""
    return _repo_root() / "Figures"
