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


def load_propeller_dict(name: str) -> list[tuple[str, Dict[str, Any]]]:
    '''Load propeller geometry dictionary from pickle file or directory.
    
    Args:
        name: Propeller name (filename without .pkl extension) or directory path
              relative to Datasets/Propellers/. If a directory is provided,
              all .pkl files in that directory will be loaded.
        
    Returns:
        List of tuples (propeller_name, geometry_dict) containing propeller
        geometry with keys: 'r', 'dr', 'chord', 'twist', 'airfoil', 'COM_shift',
        'tip_radius', 'hub_radius', 'n_blades'. Returns a single-item list if
        loading a single file.
    '''
    base_path = _repo_root() / 'Datasets'
    path = base_path / name
    
    # Check if it's a directory
    if path.is_dir():
        propellers = []
        for pkl_file in sorted(path.glob('*.pkl')):
            propeller_name = pkl_file.stem
            with pkl_file.open('rb') as f:
                propellers.append((propeller_name, pickle.load(f)))
        return propellers
    
    # Otherwise treat as file
    if not path.suffix:
        path = path.with_suffix('.pkl')
    
    # Extract name without path and extension
    propeller_name = path.stem
    with path.open('rb') as f:
        return [(propeller_name, pickle.load(f))]


def figures_dir() -> Path:
    '''Get Figures directory path.
    
    Returns:
        Path to Figures directory in repository.
    '''
    return _repo_root() / 'Figures'
