import pickle
import torch
import pyfar as pf
import numpy as np
import os
import sys

# 1. Import NVTX for labeling the timeline
import torch.cuda.nvtx as nvtx

from FastBEMT.Propeller import Propeller
from FastBEMT.JobParameters import LowFidelityParameters
from FastBEMT.TorchBPM import BPM

# Load data and setup (keep your existing setup)
with open("Datasets/Propellers/10x7E.pkl", "rb") as f:
    blade_dict = pickle.load(f)

params = LowFidelityParameters(
    rpm=7000, a_inf=343, rho=1.225, mu=1.81e-5,
    n_blades=blade_dict['n_blades'], p_ref=2e-5,
    revolutions=5, num_obs_times_per_rev=50,
)

third_octave_freqs = pf.dsp.filter.fractional_octave_frequencies(
    num_fractions=3, frequency_range=(20, 20000)
)[0]

propeller = Propeller(propeller_geometry=blade_dict, params=params)
propeller.run_bemt(v_inf=0)
r_observers = np.array([[0.1, 1.8, 0], [0.1, 0, 1.8]])

def measure():
    # Use nvtx.range_push/pop or the decorator to label segments
    nvtx.range_push("GPU_Setup")
    torch.cuda.empty_cache()
    torch.cuda.synchronize() # Sync ensures timing is accurate in Nsight
    nvtx.range_pop()
    
    nvtx.range_push("F1A_Acoustics")
    propeller.run_compact_f1a(observer_positions=r_observers, device='cuda')
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    nvtx.range_push("BPM_Calculation")
    bpm = BPM(propeller=propeller, frequencies=third_octave_freqs, device='cuda')
    bpm.run_bpm(lt=1e6)
    torch.cuda.synchronize()
    nvtx.range_pop()

if __name__ == "__main__":
    print(f"Starting Nsight-instrumented run on {torch.cuda.get_device_name(0)}...")
    
    # Warm up: The first run often includes overhead we don't want to profile
    measure() 
    
    # Actual profiled run
    # Note: We don't need the 'with profile(...)' block here; 
    # Nsight captures everything externally.
    measure()

    print("Execution finished. If running inside Nsight, stop the capture now.")