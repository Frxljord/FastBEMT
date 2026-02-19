from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Dict


def _find_repo_root(start: Path) -> Path:
    '''Locate repository root by searching for pyproject.toml.
    
    Args:
        start: Starting directory path.
        
    Returns:
        Path to repository root directory.
        
    Raises:
        FileNotFoundError: If pyproject.toml is not found in any parent directory.
    '''
    for parent in [start, *start.parents]:
        if (parent / 'pyproject.toml').exists():
            return parent
    raise FileNotFoundError('Could not find repo root (pyproject.toml).')


def _repo_root() -> Path:
    '''Get repository root directory from current working directory.
    
    Returns:
        Path to repository root.
    '''
    return _find_repo_root(Path.cwd())


def load_propeller_dict(name: str) -> Dict[str, Any]:
    '''Load propeller geometry dictionary from pickle file.
    
    Args:
        name: Propeller name (filename without .pkl extension).
        
    Returns:
        Dictionary containing propeller geometry with keys: 'r', 'dr', 'chord',
        'twist', 'airfoil', 'COM_shift', 'tip_radius', 'hub_radius', 'n_blades'.
    '''
    path = _repo_root() / 'Datasets' / 'Propellers' / f'{name}.pkl'
    with path.open('rb') as f:
        return pickle.load(f)


def figures_dir() -> Path:
    '''Get Figures directory path.
    
    Returns:
        Path to Figures directory in repository.
    '''
    return _repo_root() / 'Figures'
