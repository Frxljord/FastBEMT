import torch
import pickle
import time
import numpy as np
from FastBEMT.Propeller import Propeller
from FastBEMT.JobParameters import LowFidelityParameters

# Adjust paths as needed
blade_pkl = "Datasets/Propellers/10x7E.pkl"

print("Loading blade file:", blade_pkl)
with open(blade_pkl, 'rb') as f:
    blade_dict = pickle.load(f)

# Use smaller observer grid for quicker profiling but representative workload
x_range = np.linspace(-3.0, 3.0, 20)
y_range = np.linspace(0.0, 3.0, 12)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
r_observers = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

# Create params (match your usual settings)
params = LowFidelityParameters(
    rpm=7000,
    a_inf=343.0,
    rho=1.225,
    mu=1.81e-5,
    n_blades=blade_dict['n_blades'],
    p_ref=2e-5,
    revolutions=5,
    num_obs_times_per_rev=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

device = params.device
print("Using device:", device)

propeller = Propeller(propeller_geometry=blade_dict, params=params, dtype=torch.float32)
propeller.run_bemt(v_inf=0.0)

# Warm GPU
if device == 'cuda':
    torch.cuda.synchronize()

# Run profiler around the aeroacoustics call
activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

print("Starting profiler... this will run the aeroacoustics solver once.")
with torch.profiler.profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    with torch.profiler.record_function("run_aeroacoustics"):
        # Time and run
        start = time.time()
        propeller.run_aeroacoustics(observer_positions=r_observers)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

print(f"Aeroacoustics runtime: {end-start:.4f} s")

# Print top ops
print(prof.key_averages().table(sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total", row_limit=30))

# Save trace for TensorBoard if desired
try:
    prof.export_chrome_trace("profiler_trace.json")
    print("Saved chrome trace to profiler_trace.json")
except Exception as e:
    print("Could not save chrome trace:", e)
