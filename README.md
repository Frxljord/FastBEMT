# FastBEMT

FastBEMT is a compact Blade Element Momentum Theory solver with tonal-noise,
broadband-noise, and blade-stress utilities. NumPy/Pandas handle the aerodynamic
solution and PyTorch handles the kinematic and acoustic tensor calculations.

## Models

- BEMT section loads and integrated thrust, torque, coefficients, and figure of
  merit
- Farassat 1A compact-source thickness and loading noise
- Brooks-Pope-Marcolini broadband noise components
- Centrifugal and bending stress on the supplied airfoil sections

The maintained package lives under [`src/FastBEMT`](src/FastBEMT). Historical
implementations are available through Git history rather than being shipped in
the Python package.

## Installation

FastBEMT requires Python 3.12 or newer.

```powershell
uv sync
```

Notebook and development tools are optional:

```powershell
uv sync --extra examples
uv sync --extra dev
```

## Geometry contract

`Propeller` consumes the current PropGen-style geometry mapping. The required
entries are:

- `airfoils`: `(sections, points, 2)` normalized airfoil coordinates
- `r`, `chord`, `twist`, `sweep`, and `rake`: one value per section
- `hub_radius`, `tip_radius`, and `n_blades`

The bundled [`Data/10x7E.pkl`](Data/10x7E.pkl) is the reference contract.
Radial widths, airfoil areas, boat-tail angles, active hub-to-tip sections, and
device tensors are derived once by `Propeller`.

## Aerodynamic analysis

```python
from FastBEMT import (
    BEMT,
    Environment,
    Propeller,
    Simulation,
    load_propeller_geometry,
)

geometry = load_propeller_geometry("Data/10x7E.pkl")
environment = Environment()
simulation = Simulation(
    revolutions=2,
    timesteps_per_revolution=100,
    device="cpu",
)
propeller = Propeller(geometry, environment, simulation)

bemt = BEMT(propeller, rpm=7000, v_inf=0.0)
print(bemt.performance_for())
section_results = bemt.solution_for()
```

`rpm` and `v_inf` may also be one-dimensional arrays; FastBEMT evaluates their
Cartesian product. Supply advance ratio `J` instead of `v_inf` when that is the
natural operating-point input.

Section result names include units or physical meaning where ambiguity would
otherwise be likely, for example `inflow_angle_deg`, `section_thrust`,
`relative_velocity`, and `reynolds_number`.

## Aeroacoustics

```python
from FastBEMT import BPM, F1A
from FastBEMT.Aeroacoustics import semicircular_observer_array

observers = semicircular_observer_array(radius=2.0, n_points=19)
one_revolution = 60.0 / 7000.0

f1a = F1A(propeller, bemt)
f1a.run(
    observers,
    observer_duration=one_revolution,
    n_observer_times=100,
)

bpm = BPM(
    propeller,
    bemt,
    turbulence_length_scale=0.01,
    turbulence_intensity=5e-4,
)
bpm.run(
    observers,
    observer_duration=one_revolution,
    n_observer_times=100,
)
```

F1A also accepts the current VPM `.pt` payload through `loadings=`. That payload
contains a finite `(time, blade, section, 4)` tensor with source time in channel
zero and global-frame force per unit span in the remaining channels, plus
embedded `r_mid_m` and `dr_m` section arrays.

## Layout

- [`Aerodynamics/BEMT.py`](src/FastBEMT/Aerodynamics/BEMT.py): operating-point
  orchestration and integrated performance
- [`Aerodynamics/Section.py`](src/FastBEMT/Aerodynamics/Section.py): one-section
  aerodynamic solve
- [`Aeroacoustics/F1A.py`](src/FastBEMT/Aeroacoustics/F1A.py): tonal noise
- [`Aeroacoustics/BPM.py`](src/FastBEMT/Aeroacoustics/BPM.py): broadband noise
- [`Kinematics/Kinematics.py`](src/FastBEMT/Kinematics/Kinematics.py): rotating
  section frames and derivatives
- [`Utils/Plotter.py`](src/FastBEMT/Utils/Plotter.py): acoustic maps and stress
  plots
- [`Utils/Stress.py`](src/FastBEMT/Utils/Stress.py): structural calculations

## Validation

The maintained automated tests are under `tests`. The tracked BEMT and stress
notebooks retain higher-cost comparison workflows, while obsolete notebooks for
the removed all-in-one `Propeller.run_aeroacoustics` API have been retired.

```powershell
uv run pytest
uv run ruff check src tests
```
