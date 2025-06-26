# File: periodicity_analysis.py
import numpy as np
from scipy.signal import stft, windows
import pywt

from phase_utils import phase_slips_removal, phase_jumps_correction
from stats_utils import ttest_bf

def non_oscillatory_periodicity_spectrum(signal_duration, Fs, window_BF, method, **opts):
    """
    This function obtains the asymptotic periodicity spectrum of non-oscillatory
    activity Psi_non-oscillatory(f).

    INPUT ARGUMENTS:
    signal_duration : int or sequence of two ints
        Length (in samples). If two-element sequence, min and max lengths.
    Fs              : float
        Sampling frequency (Hz).
    window_BF       : int
        Segment length (in samples) for Bayesian periodicity spectrum.
    method          : {'STFT', 'CWT'}
        'STFT' for short-time Fourier transform; 'CWT' for continuous wavelet transform.

    Optional keyword arguments via opts dict:
    window           : int, Hann window length (samples) for STFT [default=Fs]
    overlapLength    : int, overlap length (samples) for STFT [default=Fs-1]
    FFTLength        : int, FFT length (samples) for STFT [default=Fs]
    VoicesPerOctave  : int, voices per octave for CWT [default=16]
    PhaseJumpsCorrection : bool, apply phase jumps correction [default=True]
    PhaseSlipsRemoval    : bool, apply phase slips removal [default=True]
    min_window       : int, minimum window (ms) for median filtering [default=10]
    max_window       : int, maximum window (ms) for median filtering [default=100]
    n_win            : int, number of windows for median filtering [default=10]

    OUTPUT ARGUMENTS:
    psi_non_osc_final   : ndarray, shape (freq_bins,)
        Asymptotic periodicity spectrum
    f_psi_non_osc_final : ndarray, shape (freq_bins,)
        Frequency values of psi_non_osc_final
    """
    # Validate method
    method = method.upper()
    if method not in ('STFT', 'CWT'):
        raise ValueError("Not valid. Use 'STFT' or 'CWT'.")

    # Extract optional parameters
    window = opts.get('window', Fs)
    overlapLength = opts.get('overlapLength', Fs - 1)
    FFTLength = opts.get('FFTLength', Fs)
    VoicesPerOctave = opts.get('VoicesPerOctave', 16)
    PhaseJumpsCorrection = opts.get('PhaseJumpsCorrection', True)
    PhaseSlipsRemoval = opts.get('PhaseSlipsRemoval', True)
    min_window = opts.get('min_window', 10)
    max_window = opts.get('max_window', 100)
    n_win = opts.get('n_win', 10)

    # Prepare list of signal lengths
    if np.isscalar(signal_duration):
        s_l = np.full(10, int(signal_duration))
    elif len(signal_duration) == 2:
        if signal_duration[0] > signal_duration[1]:
            raise ValueError('The first element must be lower than the second one.')
        s_l = np.round(
            np.linspace(signal_duration[0], signal_duration[1], 10)
        ).astype(int)
    else:
        raise ValueError('signal_duration cannot have more than 2 elements.')

    psi_list = []
    f_psi_non_osc_final = None

    # Loop through durations
    for length in s_l:
        # Generate 1/f noise or white noise fallback
        try:
            from colorednoise import powerlaw_psd_gaussian
            s = powerlaw_psd_gaussian(1, length)
        except ImportError:
            s = np.random.randn(length)
        s = (s - np.mean(s)) / np.std(s)
        s = s.reshape(-1, 1)

        # Frequency decomposition
        if method == 'STFT':
            f_vals, _, tfr = stft(
                s[:, 0], fs=Fs,
                window=windows.hann(window),
                nperseg=window,
                noverlap=overlapLength,
                nfft=FFTLength,
                return_onesided=True
            )
            tfr = tfr[1:-1, :]
            f_vals = f_vals[1:-1]
        else:
            tfr, f_vals = pywt.cwt(
                s[:, 0], scales=np.arange(1, FFTLength),
                wavelet='amor', sampling_period=1/Fs,
                voices_per_octave=VoicesPerOctave
            )

        # Instantaneous frequency
        phase = np.unwrap(np.angle(tfr), axis=1)
        f_inst = np.diff(phase, axis=1) / (2 * np.pi * (1/Fs))
        f_inst = f_inst.T

        # Phase corrections
        if PhaseJumpsCorrection:
            f_inst = phase_jumps_correction(f_inst, f_vals, Fs)
        if PhaseSlipsRemoval:
            f_inst = phase_slips_removal(f_inst, Fs, min_window, max_window, n_win)

        # Compute periodicity spectrum per segment
        segments = f_inst.shape[0] // window_BF
        psi_vals = np.zeros((f_inst.shape[1], segments))
        for z in range(segments):
            seg = f_inst[z*window_BF:(z+1)*window_BF, :]
            psi_vals[:, z] = -np.log(np.var(seg, axis=0))

        psi_list.append(np.mean(psi_vals, axis=1))
        if f_psi_non_osc_final is None:
            f_psi_non_osc_final = f_vals

    # Average across durations
    psi_non_osc_final = np.mean(np.stack(psi_list, axis=1), axis=1)
    return psi_non_osc_final, f_psi_non_osc_final


def periodicity_analysis(s, psi_non_osc, f_psi_non_osc, Fs, window_BF, method, **opts):
    """
    INPUT ARGUMENTS:
    s               : array-like, input signal (column vector)
    psi_non_osc     : array-like, asymptotic periodicity spectrum of non-oscillatory activity
    f_psi_non_osc   : array-like, frequency values of psi_non_osc
    Fs              : float, sampling frequency
    window_BF       : int, segment width (samples) for Bayesian periodicity spectrum
    method          : {'STFT', 'CWT'}
        Transform type

    Optional keyword arguments via opts:
    window           : int, Hann window length for STFT [default=Fs]
    overlapLength    : float, overlap length for STFT [default=Fs-1]
    FFTLength        : int, FFT length for STFT [default=Fs]
    VoicesPerOctave  : int, voices per octave for CWT [default=16]
    PhaseJumpsCorrection : bool, apply phase jumps correction [default=True]
    PhaseSlipsRemoval    : bool, apply phase slips removal [default=True]
    min_window       : int, min window (ms) for median filtering [default=10]
    max_window       : int, max window (ms) for median filtering [default=100]
    n_win            : int, number of windows for median filtering [default=10]

    OUTPUT ARGUMENTS:
    power_sp        : ndarray, power spectrum
    psi_norm        : ndarray, normalized periodicity spectrum
    bf_psi          : ndarray, Bayesian periodicity spectrum (log10)
    f               : ndarray, frequency values (Hz)
    """
    # Ensure column vectors
    s = np.asarray(s).flatten()
    psi_ref = np.asarray(psi_non_osc).flatten()
    # Default opts
    window = opts.get('window', Fs)
    overlapLength = opts.get('overlapLength', Fs-1)
    FFTLength = opts.get('FFTLength', Fs)
    VoicesPerOctave = opts.get('VoicesPerOctave', 16)
    PhaseJumpsCorrection = opts.get('PhaseJumpsCorrection', True)
    PhaseSlipsRemoval = opts.get('PhaseSlipsRemoval', True)
    min_window = opts.get('min_window', 10)
    max_window = opts.get('max_window', 100)
    n_win = opts.get('n_win', 10)

    # Frequency decomposition
    method = method.upper()
    if method == 'STFT':
        f_vals, t_vals, tfr = stft(
            s, fs=Fs,
            window=windows.hann(window),
            nperseg=window,
            noverlap=overlapLength,
            nfft=FFTLength,
            return_onesided=True
        )
        tfr = tfr[1:-1, :]
        f_vals = f_vals[1:-1]
        # Trim to reference length
        tfr = tfr[:len(psi_ref), :]
        f_vals = f_vals[:len(psi_ref)]
    elif method == 'CWT':
        tfr, f_vals = pywt.cwt(
            s, scales=np.arange(1, len(psi_ref)+1),
            wavelet='amor', sampling_period=1/Fs,
            voices_per_octave=VoicesPerOctave
        )
        # Trim any extra frequencies
        tfr = tfr[:len(psi_ref), :]
        f_vals = f_vals[:len(psi_ref)]
    else:
        raise ValueError("Not valid. Use 'STFT' or 'CWT'.")

    # 2. Instantaneous frequency
    phase = np.unwrap(np.angle(tfr), axis=1)
    f_inst = np.diff(phase, axis=1) / (2 * np.pi * (1/Fs))
    f_inst = f_inst.T

    # 3. Phase corrections
    if PhaseJumpsCorrection:
        f_inst = phase_jumps_correction(f_inst, f_vals, Fs)
    if PhaseSlipsRemoval:
        f_inst = phase_slips_removal(f_inst, Fs, min_window, max_window, n_win)

    # 5. Power spectrum per segment
    segments = f_inst.shape[0] // window_BF
    power = np.zeros((len(f_vals), segments))
    for z in range(segments):
        seg_tfr = tfr[:, z*window_BF:(z+1)*window_BF]
        if method == 'STFT':
            power[:, z] = np.mean(np.abs(seg_tfr)**2, axis=1)
        else:
            # scale factor for CWT
            FREQ = pywt.central_frequency('morl')
            scale = FREQ / f_vals
            power[:, z] = np.mean((np.abs(seg_tfr)**2) * np.sqrt(scale)[:, None], axis=1)
    power_sp = np.mean(power, axis=1)

    # 6. Normalized periodicity spectrum
    psi = np.zeros((len(f_vals), segments))
    for z in range(segments):
        seg = f_inst[z*window_BF:(z+1)*window_BF, :]
        psi[:, z] = -np.log(np.var(seg, axis=0)) - psi_ref
    psi_norm = np.mean(psi, axis=1)

    # 7. Bayesian periodicity spectrum
    bf_psi = np.zeros(len(f_vals))
    for z in range(len(f_vals)):
        bf_val, _ = ttest_bf(psi[z, :], tail='right')
        bf_psi[z] = np.log10(bf_val)

    return power_sp, psi_norm, bf_psi, f_vals
