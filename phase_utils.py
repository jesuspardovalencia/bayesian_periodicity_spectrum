# File: phase_utils.py
import numpy as np
from scipy.signal import medfilt

def phase_slips_removal(f_inst, Fs, min_window, max_window, n_win):
    """
    Based on "Fluctuations in Oscillation Frequency Control Spike Timing
    and Coordinate Neural Networks", published in The Journal of
    Neuroscience (https://doi.org/10.1523/JNEUROSCI.0261-14.2014).

    INPUT ARGUMENTS:
    f_inst     : array-like, shape (time, frequencies)
                 Instantaneous frequency matrix
    Fs         : float
                 Sampling frequency (Hz)
    min_window : float
                 Minimum window length (in ms) for median filtering
    max_window : float
                 Maximum window length (in ms) for median filtering
    n_win      : int
                 Number of windows for median filtering

    OUTPUT ARGUMENT:
    f_inst_final : ndarray, shape (time, frequencies)
                   Corrected frequency matrix after slip removal
    """
    # Create linearly spaced window lengths (ms)
    windows = np.linspace(min_window, max_window, n_win)
    time_pts, n_freq = f_inst.shape
    # Preallocate filtered array
    f_inst_filtered = np.empty((time_pts, n_freq, n_win))

    # Apply moving median for each window
    for j, win_ms in enumerate(windows):
        # Convert window length from ms to samples
        kernel_size = int(round(win_ms * Fs / 1000))
        # Kernel size must be odd and >=1
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Apply median filter along time axis with kernel (kernel_size, 1)
        f_inst_filtered[:, :, j] = medfilt(f_inst, kernel_size=(kernel_size, 1))

    # Compute median across all filtered windows
    f_inst_final = np.median(f_inst_filtered, axis=2)
    return f_inst_final

def phase_jumps_correction(f_inst, f, Fs):
    """
    This function corrects jumps of +- 2 * pi in the instantaneous
    frequency traces.

    INPUT ARGUMENTS:
    f_inst : array-like, shape (time, frequencies)
        Instantaneous frequency matrix
    f       : array-like
        Frequencies array (Hz)
    Fs      : float
        Sampling frequency (Hz)

    OUTPUT ARGUMENT:
    f_inst_final : ndarray, shape (time, frequencies)
        Corrected frequency matrix
    """
    # Initialize output array
    f_inst_final = np.zeros_like(f_inst)
    threshold = Fs / 2

    # Loop over frequency columns
    for z in range(f_inst.shape[1]):
        f_inst_t = f_inst[:, z].copy()
        # Identify indices where jump exceeds threshold
        jumps = np.where(np.abs(np.diff(f_inst_t)) > threshold)[0]

        # Correct each jump segment
        for j in jumps:
            idx = slice(j + 1, None)
            if f_inst_t[j + 1] > f_inst_t[j]:
                f_inst_t[idx] -= 2 * threshold
            else:
                f_inst_t[idx] += 2 * threshold

            # Check segment mean and adjust if beyond expected range
            segment_mean = f_inst_t[idx].mean()
            if segment_mean < f[z] - 2 * threshold:
                f_inst_t[idx] += 2 * threshold
            elif segment_mean > f[z] + 2 * threshold:
                f_inst_t[idx] -= 2 * threshold

        # Final boundary corrections
        idx_high = f_inst_t > f[z] + threshold
        f_inst_t[idx_high] -= 2 * threshold
        idx_low = f_inst_t < f[z] - threshold
        f_inst_t[idx_low] += 2 * threshold

        # Store corrected trace
        f_inst_final[:, z] = f_inst_t

    return f_inst_final
 