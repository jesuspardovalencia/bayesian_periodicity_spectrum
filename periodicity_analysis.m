function [power_sp, psi_norm, bf_psi, f] = periodicity_analysis(s, psi_non_osc, f_psi_non_osc, Fs, window_BF, method, opts)
    %% Input arguments
    % s: Input signal (column vector)
    % psi_non_osc: Asymptotic periodicity spectrum of non-oscillatory
    % activity 𝛹𝑛𝑜𝑛−𝑜𝑠𝑐𝑖𝑙𝑙𝑎𝑡𝑜𝑟𝑦(𝑓) (column vector).
    % f_psi_non_osc: Frequency values of psi_non_osc 
    % Fs: Sampling frequency
    % window_BF: Width of the segments (in samples) in which the input
    % signal is going to be divided for posterior statistics (Bayesian periodicity spectrum).
    % transform ('CWT').
    
    % Optional:
    % window: For STFT. Length (in samples) of the Hann window.  
    % overlapLength: For STFT. Overlapping (in samples) of the Hann window.
    % FFTLength: For STFT. Length (in samples) of FFT.
    % VoicesPerOctave: For CWT. Voices per octave.
    % PhaseJumpsCorrection: True or false.
    % PhaseSlipsRemoval: True or false.
    % min_window: Minimum window length (in ms) for median filtering
    % (PhaseSlipsRemoval).
    % max_window: Maximum window length (in ms) for median filtering
    % (PhaseSlipsRemoval).
    % n_win: Number of windows for median filtering (PhaseSlipsRemoval).
    
    % Example:
    % [power, psi, bf, f] = periodicity_analysis(signal, psi_ref, f_psi_ref, 1000, 2, "STFT", ...
    % overlapLength=0.5, ...
    % FFTLength=2048, ...
    % VoicesPerOctave=32, ...
    % PhaseSlipsRemoval=false);
    
    %% Output arguments
    % power_sp: Power spectrum
    % psi_norm: Normalized periodicity spectrum
    % bf_psi: Bayesian periodicity spectrum
    % f: Vector containing frequency values (Hz)
    
    arguments
        s
        psi_non_osc
        f_psi_non_osc
        Fs
        window_BF
        method {mustBeMember(method, ["STFT", "CWT"])}
        opts.window (1,1) double {mustBePositive, mustBeInteger} = Fs
        opts.overlapLength (1,1) double {mustBeNonnegative} = Fs-1
        opts.FFTLength (1,1) double {mustBePositive, mustBeInteger} = Fs
        opts.VoicesPerOctave (1,1) double {mustBePositive, mustBeInteger} = 16
        opts.PhaseJumpsCorrection (1,1) logical = true
        opts.PhaseSlipsRemoval (1,1) logical = true
        opts.min_window (1,1) double {mustBePositive, mustBeInteger} = 10
        opts.max_window (1,1) double {mustBePositive, mustBeInteger} = 100
        opts.n_win (1,1) double {mustBePositive, mustBeInteger} = 10
    end
    
    %% Variables in columns
    if size(s,1) == 1
        s = s';
    end
    if size(psi_non_osc,1) == 1
        psi_non_osc = psi_non_osc';
    end
    
    %% 1. Frequency decomposition
    if strcmp(method, 'STFT')
        [tfr, f, ~] = stft(s, Fs, 'Window', hann(opts.window), 'OverlapLength', opts.overlapLength, 'FFTLength', opts.FFTLength, 'FrequencyRange', 'onesided');
        tfr([1,end],:) = []; f([1,end]) = []; % Remove 0 Hz and Nyquist frequency
        tfr(length(f_psi_non_osc)+1:end,:) = [];
        f(length(f_psi_non_osc)+1:end) = [];
    elseif strcmp(method, 'CWT')
        [tfr, f] = cwt(s, 'amor', Fs, 'voicesPerOctave', opts.VoicesPerOctave);
        tfr(length(f_psi_non_osc)+1:end,:) = [];
        f(length(f_psi_non_osc)+1:end) = [];
        FREQ = centfrq('morl');
        scale = []; scale = (FREQ./f);
    else
        error('Not valid. Use STFT or CWT.');
    end
    
    %% 2. Instantaneous frequency
    for j1 = 1:1:size(tfr,1)
        f_inst(j1,:) = diff(unwrap(angle(tfr(j1,:)))) / (2 * pi * (1 / Fs));
    end
    
    %% 3. Phase jumps correction
    if opts.PhaseJumpsCorrection
        f_inst_corrected = phase_jumps_correction(f_inst', f, Fs);
    else
        f_inst_corrected = f_inst';
    end
    
    %% 4. Phase slips removal
    if opts.PhaseSlipsRemoval
        f_inst_final = phase_slips_removal(f_inst_corrected, Fs, opts.min_window, opts.max_window, opts.n_win);
    else
        f_inst_final = f_inst_corrected;
    end
    
    %% 5. Power spectrum
    segments = floor(size(f_inst_final,1)/(window_BF));
    for z = 1:1:segments
        if strcmp(method, 'STFT')
            power(:,z) = mean(abs(tfr(:,(z-1)*(window_BF)+1:z*((window_BF)))).^2, 2)'; % Measured as the inverse logarithm of the variance of the instantaneous frequency
        elseif strcmp(method, 'CWT')
            power(:,z) = mean(abs(tfr(:,(z-1)*(window_BF)+1:z*((window_BF)))).^2 .* sqrt(scale), 2)'; % Measured as the inverse logarithm of the variance of the instantaneous frequency
        end
    end
    power_sp = mean(power,2);
    
    %% 6. Normalized periodicity spectrum
    for z = 1:1:segments
        psi(:,z) = -log(var(f_inst_final((z-1)*(window_BF)+1:z*((window_BF)),:))) - psi_non_osc'; % Measured as the inverse logarithm of the variance of the instantaneous frequency
    end
    psi_norm = mean(psi,2);
    
    %% 7. Bayesian periodicity spectrum
    % This function includes code from Bart Krekelberg (2025). bayesFactor
    % (https://github.com/klabhub/bayesFactor), GitHub. Retrieved March 3,
    % 2025.
    for z = 1:1:length(f)
        [bf_psi(z), ~] = ttest_bf(psi(z,:),'tail','right');
        bf_psi(z) = log10(bf_psi(z));
    end
    
end
