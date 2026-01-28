import pickle
import numpy as np
import pyfar as pf
import timeit
from FastBEMT.Propeller import Propeller
from FastBEMT.JobParameters import AerodynamicParameters, AcousticParameters
from FastBEMT.BPM import BPM

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
    revolutions=100,
    num_obs_times_per_rev=100
)

third_octave_freqs = pf.dsp.filter.fractional_octave_frequencies(num_fractions=3, frequency_range=(20,20000))[0]

n_iterations = 20

def measure():
    propeller = Propeller(
        propeller_geometry=blade_dict,
        params=aerodynamic_params,
        params=acoustic_params
    )
    propeller.run_bemt(v_inf=0)

    # r_observer = np.array([[0.1, 1.8, 0]])
    # propeller.run_compact_f1a(observer_positions=r_observer, use_GPU=True)
    # bpm = BPM(propeller=propeller, frequencies = third_octave_freqs)
    # bpm.run_BPM()

    # F1A_spl_3oct = []
    # for fc in third_octave_freqs:
    #     f_low, f_high = fc / (2**(1/6)), fc * (2**(1/6))
    #     mask = (propeller.freq[:, 0] >= f_low) & (propeller.freq[:, 0] < f_high)
    #     if np.any(mask):
    #         P_band = np.sum(10**(propeller.spl[mask] / 10))
    #         F1A_spl_3oct.append(10 * np.log10(P_band))
    #     else:
    #         F1A_spl_3oct.append(-np.inf)

    # F1A_spl_3oct = np.array(F1A_spl_3oct)

    # L_total = 10 * np.log10(
    #     10**(F1A_spl_3oct / 10) + 
    #     10**(bpm.SPL_LBL / 10) + 
    #     10**(bpm.SPL_TBL / 10) + 
    #     10**(bpm.SPL_TEB / 10) + 
    #     10**(bpm.SPL_TI / 10) + 
    #     10**(bpm.SPL_TIP / 10)
    # )

    # return 10 * np.log10(np.sum(10**(L_total / 10)))

measure()

# avg_time = timeit.timeit(measure, number=n_iterations) / n_iterations

# print(f"Average BEMT execution time over {n_iterations} runs: {avg_time:.4f} seconds")