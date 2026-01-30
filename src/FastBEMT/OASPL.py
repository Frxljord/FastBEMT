import numpy as np


def calculate_OASPL(t, p, p_ref=2e-5):
    n = len(t)
    n = t.shape[0]
    dt = t[1] - t[0]
    freq = np.fft.rfftfreq(n, dt)

    fft_p = np.fft.rfft(p, axis=0)
    fft_amp = np.abs(fft_p) * np.sqrt(2) / n
    fft_amp[0] /= np.sqrt(2)

    spl = 20 * np.log10(np.maximum(fft_amp, 1e-15) / p_ref)

    def r_a_func(f):
        f_sq = f**2
        return (12194**2 * f_sq**2) / (
            (f_sq + 20.6**2)
            * np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2))
            * (f_sq + 12194**2)
        )

    a_weight = 20 * np.log10(r_a_func(freq)) - 20 * np.log10(r_a_func(1000))
    spl_a = spl + a_weight
    amp_a = 10 ** (spl_a / 20) * p_ref
    p_rms_a = np.sqrt(amp_a[0] ** 2 + 2 * np.sum(amp_a[1:] ** 2, axis=0))
    return 20 * np.log10(p_rms_a / p_ref)
