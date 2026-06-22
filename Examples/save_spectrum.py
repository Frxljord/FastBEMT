import numpy as np
import pandas as pd
from pathlib import Path

# Get the current notebook's kernel variables
# (This script should be run after executing the VPM-F1A notebook)

# Assuming the variables are already in the kernel from notebook execution
# We'll save them to CSV files

output_dir = Path(__file__).parent / "spectrum_data"
output_dir.mkdir(exist_ok=True)

# Get data from kernel (these variables must exist from notebook execution)
import IPython
kernel = IPython.get_ipython()

if kernel is not None:
    # Get variables from the kernel
    vpm_frequency = kernel.user_ns.get('vpm_frequency')
    vpm_spl = kernel.user_ns.get('vpm_spl')
    bemt_frequency = kernel.user_ns.get('bemt_frequency')
    bemt_spl = kernel.user_ns.get('bemt_spl')
    blade_passing_frequency = kernel.user_ns.get('blade_passing_frequency')
    
    if all([vpm_frequency is not None, vpm_spl is not None, 
            bemt_frequency is not None, bemt_spl is not None]):
        
        # Create DataFrames for VPM spectrum
        vpm_df = pd.DataFrame({
            'Frequency_Hz': vpm_frequency,
            'SPL_dB': vpm_spl,
            'BPF_Ratio': vpm_frequency / blade_passing_frequency
        })
        
        # Create DataFrames for BEMT spectrum
        bemt_df = pd.DataFrame({
            'Frequency_Hz': bemt_frequency,
            'SPL_dB': bemt_spl,
            'BPF_Ratio': bemt_frequency / blade_passing_frequency
        })
        
        # Save to CSV files
        vpm_csv = output_dir / "vpm_spectrum.csv"
        bemt_csv = output_dir / "bemt_spectrum.csv"
        
        vpm_df.to_csv(vpm_csv, index=False)
        bemt_df.to_csv(bemt_csv, index=False)
        
        print(f"VPM spectrum saved to: {vpm_csv}")
        print(f"BEMT spectrum saved to: {bemt_csv}")
        
        # Also save combined data
        combined_csv = output_dir / "bemt_vpm_spectrum_comparison.csv"
        combined_df = pd.DataFrame({
            'Frequency_Hz': bemt_frequency,
            'BEMT_SPL_dB': bemt_spl,
            'VPM_SPL_dB': vpm_spl,
            'BPF_Ratio': bemt_frequency / blade_passing_frequency
        })
        combined_df.to_csv(combined_csv, index=False)
        print(f"Combined spectrum saved to: {combined_csv}")
    else:
        print("Error: Required variables not found in kernel. Please run the VPM-F1A notebook first.")
else:
    print("This script must be run from a Jupyter notebook kernel.")
