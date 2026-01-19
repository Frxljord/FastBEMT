import pickle
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from Propeller import Propeller
from JobParameters import AerodynamicParameters, AcousticParameters

with open("Datasets/Propellers/10x7E.pkl", "rb") as f:
    blade_dict = pickle.load(f)

aerodynamic_params = AerodynamicParameters(
    prop_radius=blade_dict['tip_radius'],
    hub_radius=blade_dict['hub_radius'],
    n_blades=blade_dict['n_blades'],
    rpm=7000,
    a_inf=343,
    rho=1.225,
    mu=1.81e-5,
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

propeller.run_bemt(v_inf=0)

r_observer = np.array([[0, 1.88, 0]])
propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=True)

def run_performance_test(scales, r_observer):
    cpu_results = []
    gpu_results = []
    
    for size in scales:
        print(f"Testing scale: {size} points...")

        acoustic_params = AcousticParameters(
            aero_params=aerodynamic_params,
            p_ref=2e-5, revolutions=size//100, num_obs_times_per_rev=100
        )
        propeller = Propeller(
            propeller_geometry=blade_dict,
            aero_params=aerodynamic_params,
            acoustic_params=acoustic_params
        )
        propeller.run_bemt(v_inf=0)

                
        # --- GPU BENCHMARK ---
        # Warm-up
        propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=True)
        torch.cuda.synchronize()
        
        start_gpu = torch.cuda.Event(enable_timing=True)
        end_gpu = torch.cuda.Event(enable_timing=True)
        
        start_gpu.record()
        propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=True)
        end_gpu.record()
        torch.cuda.synchronize()
        gpu_results.append(start_gpu.elapsed_time(end_gpu)) # ms

        # --- CPU BENCHMARK ---
        # Warm-up
        propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=False)

        start_cpu = time.perf_counter()
        propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=False)
        end_cpu = time.perf_counter()
        cpu_results.append((end_cpu - start_cpu) * 1000)

    return cpu_results, gpu_results

problem_scales = [1000, 5000, 10000, 50000, 100000]

cpu_times, gpu_times = run_performance_test(problem_scales, r_observer)

plt.figure(figsize=(10, 6))
plt.plot(problem_scales, cpu_times, 'o-', label='CPU Execution', color='tab:red', linewidth=2)
plt.plot(problem_scales, gpu_times, 's-', label='GPU Execution', color='tab:blue', linewidth=2)

plt.xlabel('Simulation Time Steps', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plt.show()