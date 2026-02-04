import pickle
import cProfile
import pstats
import subprocess
from FastBEMT.Propeller import Propeller
from FastBEMT.JobParameters import LowFidelityParameters

with open("Datasets/Propellers/10x7E.pkl", "rb") as f:
    blade_dict = pickle.load(f)

params = LowFidelityParameters(
    rpm=7000,
    a_inf=343,
    rho=1.225,
    mu=1.81e-5,
    n_blades=blade_dict['n_blades'],
    p_ref=2e-5,
    revolutions=5,
    num_obs_times_per_rev=50,
)

def measure():
    propeller = Propeller(
        propeller_geometry=blade_dict,
        params=params,
    )
    propeller.run_bemt(v_inf=0)

if __name__ == "__main__":
    prof_file = "program.prof"    
    profiler = cProfile.Profile()
    profiler.enable()
    measure()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(prof_file)
    print(f"Launching SnakeViz for {prof_file}...")
    try:
        subprocess.run(["snakeviz", prof_file])
    except FileNotFoundError:
        print("Error: SnakeViz not found. Please run 'pip install snakeviz' first.")