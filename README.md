# FastBEMT

A fast approximate BEMT (Blade Element Momentum Theory) solver for propeller aeroacoustic analysis.

## Overview

FastBEMT provides a high-performance Python package for analyzing propeller aerodynamics and acoustics using Blade Element Momentum Theory combined with acoustic source models. The package leverages PyTorch for GPU acceleration, enabling efficient computation of:

- **Aerodynamic Analysis**: Blade Element Momentum Theory (BEMT) calculations with Prandtl loss corrections
- **Tonal Noise**: Farassat 1A compact source formulation for thickness and loading noise
- **Broadband Noise**: Brooks-Pope-Marcolini (BPM) model for broadband noise prediction
- **Structural Analysis**: Blade stress calculations under centrifugal and bending loads

## Package Structure

### src/FastBEMT

#### [Propeller.py](src/FastBEMT/Propeller.py)
Main propeller analysis class orchestrating the complete aeroacoustic workflow. Handles:
- Initialization of propeller geometry and simulation parameters
- BEMT aerodynamic solution across blade sections
- F1A acoustic source computation (monopole and dipole noise)
- BPM broadband noise prediction in third-octave bands
- Integration of results into output acoustic spectra and time histories

#### [Section.py](src/FastBEMT/Section.py)
Blade Element Momentum Theory solver for individual propeller sections. Provides:
- Iterative solution of momentum and blade element equations
- Prandtl tip and hub loss factor computations
- Airfoil coefficient interpolation using pre-built aerosandbox.Airfoil objects
- Mach and Reynolds number effects on aerodynamic coefficients
- Local inflow angle and force distribution calculations

#### [BPM.py](src/FastBEMT/BPM.py)
Brooks-Pope-Marcolini broadband noise prediction model (PyTorch implementation). Implements five distinct noise sources:
- **Turbulent Boundary Layer (TBL)**: Noise from turbulent pressure fluctuations on blade surfaces including suction-side, pressure-side, and separated-flow components
- **Laminar Boundary Layer (LBL)**: Instability noise from laminar boundary layers at low frequencies
- **Trailing Edge Bluntness (TEB)**: Scattering of incoming vorticity by blunt trailing edges
- **Tip Vortex (TV)**: Noise from unsteady loading fluctuations induced by tip vortex
- **Turbulence Ingestion (TI)**: Interaction of ingested turbulence with blade surfaces

Features include GPU-accelerated Strouhal number and correction factor calculations, and third-octave band spectrum generation.

#### [F1A.py](src/FastBEMT/F1A.py)
Farassat 1A acoustic formulation (PyTorch implementation) for rotating sources. Handles:
- Thickness (monopole) source noise from blade volume displacement
- Loading (dipole) source noise from aerodynamic forces
- Compact source approximation for efficient far-field calculation
- GPU-accelerated tensor operations for time-domain pressure computation
- Observer position and blade angle handling

#### [JobParameters.py](src/FastBEMT/JobParameters.py)
Simulation parameter container storing:
- Operational parameters (RPM, angular velocity)
- Environmental properties (air density, speed of sound, dynamic viscosity)
- Acoustic reference properties (reference pressure)
- Time discretization (revolutions, time steps per revolution, observer time range)
- PyTorch device specification (CPU or CUDA GPU)

#### [DataLoader.py](src/FastBEMT/DataLoader.py)
Utility functions for data input/output:
- Loading propeller geometry dictionaries from pickle files in the Datasets directory
- Repository root detection and path management
- Access to project figure output directories

#### [Plotter.py](src/FastBEMT/Plotter.py)
Visualization utilities for acoustic analysis results:
- Time-domain pressure histories for monopole, dipole, and total pressure
- Frequency-domain acoustic spectra (Sound Pressure Level)
- Blade passing frequency harmonic indicators
- Overall A-weighted Sound Pressure Level (OASPL) display
- Multi-observer comparison plots

#### [Stress.py](src/FastBEMT/Stress.py)
Blade structural analysis tools computing:
- Centrifugal stress distribution along blade span
- Bending stress from thrust and torque loads
- Moment of inertia calculations for arbitrary airfoil sections
- Combined stress field at each blade section

## Requirements

- Python ≥ 3.12
- PyTorch (for GPU acceleration)
- NumPy, SciPy (numerical computing)
- AeroSandbox (airfoil aerodynamics)
- Plotly, Matplotlib (visualization)
- scikit-learn (data processing)
- And additional dependencies as specified in pyproject.toml

## Installation

```bash
pip install -e .
```
