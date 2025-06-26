# Bayesian periodicity spectrum
Toolbox (MATLAB, Python) for the Bayesian periodicity spectrum.

This code was developed for the computation of the Bayesian periodicity spectrum, which is based on the variability of the instantaneous frequency. The Bayesian periodicity spectrum quantifies oscillatory activity at each frequency component without confounds from underlying non-oscillatory components, and is extended with a Bayesian framework to formally assess the presence or the absence of oscillations. With this novel approach, we aim to resolve the ambiguity of algorithms that rely solely on the power spectrum for the separation of oscillatory and non-oscillatory components in signals.
# Basic usage
Given a signal of interest s(t), the "Bayesian periodicity spectrum toolbox" first calculates the corresponding periodicity of non-oscilaltory components, depending of the method used for the frequency component decomposition — for this toolbox, the short-time Fourier transform (STFT) or the continuous wavelet transform (CWT). If multiple signals with different durations share the same decomposition parameters, use the shortest signal duration to compute the periodicity of the non-oscillatory components. This reduces computation time — shorter simulated signals require fewer resources — and the resulting periodicity can then be applied to all signals of interest. For the preprocessing of the instantaneous frequency, it is optional to correct for jumps of +-2*pi*Fs and to reduce phase slips in the instantaneous frequency traces, although it is recommended. The periodicity spectrum is estimated as the inverse logarithm of the variance of the instantaneous frequency for each component. Instantaneous frequency traces are divided into segments of equal length in order to estimate the Bayesian periodicity spectrum. This strategy allows to assess the level of evidence for the presence or the absence of oscillatory activity in a single signal.

Below there is a typical workflow to obtain the power, periodicity, and Bayesian periodicity spectra of a signal in MATLAB. This workflow is equivalent in Python.

```matlab
% ---------------------------------------------------------------------
% In this example the signal is stored in a variable called 's', 
% a M x 1 array.
% The sampling frequency is set to 128 Hz, the segments in which the
% time series will be divided is set to a length of 128 samples, and
% the decomposition method is the continuous wavelet transform.
% Optional input arguments are set to default values.
% ---------------------------------------------------------------------
Fs = 128;
window = Fs;
method = 'CWT';

% ------------- calculate the non-oscillatory periodicity -------------
[psi_non_osc_final, f_psi_non_osc_final] = non_oscillatory_periodicity_spectrum(length(s), Fs, window, method);

% ------------- calculate power and (Bayesian) periodicity spectra ----
power_sp = []; psi_norm = []; bf_psi = []; f = [];
[power_sp, psi_norm, bf_psi, f] = periodicity_analysis(s,...
    psi_non_osc_final, f_psi_non_osc_final, Fs, window, method);
```

In order to compare the power, periodicity or Bayesian periodicity between two populations y1 and y2, there is a function called 'spectra_comparison' made for this purpose. This function performs a Bayesian paired t-test for each frequency component.
