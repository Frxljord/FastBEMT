import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from FastBEMT.JobParameters import LowFidelityParameters
from FastBEMT.Propeller import Propeller

with open('Datasets/Propellers/10x7E.pkl', 'rb') as f:
    blade_dict = pickle.load(f)


def run_performance_test(scales: list[int], r_observer: np.ndarray) -> tuple[list[float], list[float]]:
    cpu_results: list[float] = []
    gpu_results: list[float] = []

    for size in scales:
        print(f'Testing scale: {size} time steps...')

        revolutions = max(1, size // 100)

        # --- GPU BENCHMARK ---
        if torch.cuda.is_available():
            params_gpu = LowFidelityParameters(
                rpm=7000,
                a_inf=343,
                rho=1.225,
                mu=1.81e-5,
                n_blades=blade_dict['n_blades'],
                p_ref=2e-5,
                revolutions=revolutions,
                num_obs_times_per_rev=100,
                device='cuda',
            )
            propeller_gpu = Propeller(
                geometry=blade_dict,
                params=params_gpu,
                use_cuda_timing=False,
            )
            propeller_gpu.run_bemt(v_inf=0)

            # Warm-up
            propeller_gpu.run_aeroacoustics(observer_positions=r_observer)
            torch.cuda.synchronize()

            start_gpu = torch.cuda.Event(enable_timing=True)
            end_gpu = torch.cuda.Event(enable_timing=True)
            start_gpu.record()
            propeller_gpu.run_aeroacoustics(observer_positions=r_observer)
            end_gpu.record()
            torch.cuda.synchronize()
            gpu_results.append(start_gpu.elapsed_time(end_gpu))
        else:
            gpu_results.append(float('nan'))

        # --- CPU BENCHMARK ---
        params_cpu = LowFidelityParameters(
            rpm=7000,
            a_inf=343,
            rho=1.225,
            mu=1.81e-5,
            n_blades=blade_dict['n_blades'],
            p_ref=2e-5,
            revolutions=revolutions,
            num_obs_times_per_rev=100,
            device='cpu',
        )
        propeller_cpu = Propeller(
            geometry=blade_dict,
            params=params_cpu,
            use_cuda_timing=False,
        )
        propeller_cpu.run_bemt(v_inf=0)

        # Warm-up
        propeller_cpu.run_aeroacoustics(observer_positions=r_observer)

        start_cpu = time.perf_counter()
        propeller_cpu.run_aeroacoustics(observer_positions=r_observer)
        end_cpu = time.perf_counter()
        cpu_results.append((end_cpu - start_cpu) * 1000.0)

    return cpu_results, gpu_results


observer_positions = np.array([[0.0, 1.88, 0.0]])
problem_scales = [100, 500, 1000, 5000, 10000, 50000]

cpu_times, gpu_times = run_performance_test(problem_scales, observer_positions)

plt.figure(figsize=(10, 6))
plt.plot(
    problem_scales,
    cpu_times,
    'o-',
    label='CPU Execution',
    color='tab:red',
    linewidth=2,
)
plt.plot(
    problem_scales,
    gpu_times,
    's-',
    label='GPU Execution',
    color='tab:blue',
    linewidth=2,
)

plt.xlabel('Simulation Time Steps', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', ls='-', alpha=0.5)
plt.legend()

plt.show()