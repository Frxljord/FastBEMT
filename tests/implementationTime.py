import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.Propeller import Propeller
from src.JobParameters import AerodynamicParameters, AcousticParameters

with open("10x7E.pkl", "rb") as f:
    blade_dict = pickle.load(f)

aerodynamic_params = AerodynamicParameters(
    prop_radius=blade_dict['tip_radius'],
    hub_radius=blade_dict['hub_radius'],
    n_blades=blade_dict['n_blades'],
    rpm=7000,
    v_inf=0,
    a_inf=343,
    rho=1.225,
    mu=1.81e-5,
    p_ref=2e-5
)

acoustic_params = AcousticParameters(
    aero_params=aerodynamic_params,
    p_ref=2e-5,
    revolutions=10,
    num_obs_times_per_rev=100
)

propeller = Propeller(
    propeller_geometry=blade_dict,
    aero_params=aerodynamic_params,
    acoustic_params=acoustic_params
)

import timeit

n_iterations = 20

def measure():
    propeller.run_bemt()

avg_time_1 = timeit.timeit(measure, number=n_iterations) / n_iterations

print(f"Average BEMT execution time over {n_iterations} runs: {avg_time_1:.4f} seconds")