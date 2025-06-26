function [psi_non_osc_final, f_psi_non_osc_final] = non_oscillatory_periodicity_spectrum(signal_duration, Fs, window_BF, method, opts)
    % This function obtains the asymptotic periodicity spectrum of non-oscillatory activity 𝛹𝑛𝑜𝑛−𝑜𝑠𝑐𝑖𝑙𝑙𝑎𝑡𝑜𝑟𝑦(𝑓)
    
    %% Input arguments
    % signal_duration: Length (in samples). If there is more than one
    % signal, signal_duration is an array containing the minimum and maximum length of the signals of interest.
    % Fs: Sampling frequency.
    % window_BF: Length of the segments (in samples) in which the input
    % signal is going to be divided for posterior statistics (Bayesian periodicity spectrum).
    % method: Short-time Fourier transform ('STFT') or Continuous wavelet
    % transform ('CWT').
    
    % Optional:
    % window: For STFT. Length (in samples) of the Hann window.  
    % overlapLength: For STFT. Overlapping (in samples) of the Hann window.
    % FFTLength: For STFT. Length (in samples) of FFT.
    % VoicesPerOctave: For CWT. Voices per octave.
    % PhaseSlipsRemoval: True or false.
    % min_window: Minimum window length (in ms) for median filtering
    % (PhaseSlipsRemoval).
    % max_window: Maximum window length (in ms) for median filtering
    % (PhaseSlipsRemoval).
    % n_win: Number of windows for median filtering (PhaseSlipsRemoval).
    
    % Example:
    % [psi, f] = non_oscillatory_periodicity_spectrum([3000, 4000], 1000, 'STFT', ...
    % window=3, ...
    % overlapLength=1.5, ...
    % FFTLength=4096, ...
    % VoicesPerOctave=48, ...
    % PhaseSlipsRemoval=false);
    
    %% Output arguments
    % psi_non_osc: Asymptotic periodicity spectrum of non-oscillatory activity 𝛹𝑛𝑜𝑛−𝑜𝑠𝑐𝑖𝑙𝑙𝑎𝑡𝑜𝑟𝑦(𝑓) (column vector)
    % f_psi_non_osc: Frequency values of psi_non_osc 
    
    arguments
        signal_duration
        Fs
        window_BF
        method {mustBeMember(method, ["STFT", "CWT"])}
        opts.window (1,1) double {mustBePositive, mustBeInteger} = Fs
        opts.overlapLength (1,1) double {mustBePositive, mustBeInteger} = Fs-1
        opts.FFTLength (1,1) double {mustBePositive, mustBeInteger} = Fs
        opts.VoicesPerOctave (1,1) double {mustBePositive, mustBeInteger} = 16
        opts.PhaseJumpsCorrection (1,1) logical = true
        opts.PhaseSlipsRemoval (1,1) logical = true
        opts.min_window (1,1) double {mustBePositive, mustBeInteger} = 10
        opts.max_window (1,1) double {mustBePositive, mustBeInteger} = 100
        opts.n_win (1,1) double {mustBePositive, mustBeInteger} = 10
    end
    
    rng('shuffle');
    
    if strcmp(method, 'STFT')
    elseif strcmp(method, 'CWT')
    else
        error('Not valid. Use STFT or CWT.');
    end
    % We calculate the periodicity of 1000 non-oscillatory signals
    if length(signal_duration) == 1
        s_l = round(ones(10,1)*signal_duration);
    elseif length(signal_duration) == 2
        if signal_duration(1) > signal_duration(2)
            error('The first element must be lower than the second one.')
        end
        s_l = round(signal_duration(1):(signal_duration(2)-signal_duration(1))/9:signal_duration(2));
    else
        error('"signal_duration" cannot have more than 2 elements.')
    end
    
    non_osc = []; psi_non_osc = [];
    for j = 1:1:length(s_l)
        cn = dsp.ColoredNoise(1,s_l(j),100); % 1/f noise, InverseFrequencyPower is set to 1 by default
        s = cn();
        s = (s - mean(s)) ./ std(s);

        for i = 1:1:size(s,2)
            %% 1. Frequency decomposition
            if strcmp(method, 'STFT')
                [tfr, f, ~] = stft(s(:,i), Fs, 'Window', hann(opts.window), 'OverlapLength', opts.overlapLength, 'FFTLength', opts.FFTLength, 'FrequencyRange', 'onesided');
                tfr([1,end],:) = []; f([1,end]) = []; % Remove 0 Hz and Nyquist frequency
            elseif strcmp(method, 'CWT')
                [tfr, f] = cwt(s(:,i), 'amor', Fs, 'voicesPerOctave', opts.VoicesPerOctave);
            end
            
            %% 2. Instantaneous frequency
            f_inst = [];
            for j1 = 1:1:size(tfr,1)
                f_inst(j1,:) = diff(unwrap(angle(tfr(j1,:)))) / (2 * pi * (1 / Fs));
            end
            f_inst = f_inst';

            %% 3. Phase jumps correction
            if opts.PhaseJumpsCorrection
                f_inst_corrected = phase_jumps_correction(f_inst, f, Fs);
            else
                f_inst_corrected = f_inst;
            end

            %% 4. Phase slips removal
            if opts.PhaseSlipsRemoval
                f_inst_final = phase_slips_removal(f_inst_corrected, Fs, opts.min_window, opts.max_window, opts.n_win);
            else
                f_inst_final = f_inst_corrected;
            end

            %% 5. Periodicity
            segments = floor(size(f_inst_final,1)/(window_BF));
            for z = 1:1:segments
                psi = -log(var(f_inst_final((z-1)*(window_BF)+1:z*((window_BF)),:))); % Measured as the inverse logarithm of the variance of the instantaneous frequency
                non_osc(j).value(:,i,z) = psi;
            end

        end
        if j > 1
            lim = length(psi_non_osc(:,1));
            psi_non_osc(:,j) = mean(mean(non_osc(j).value(1:lim,:),3),2);
        else
            psi_non_osc(:,j) = mean(mean(non_osc(j).value,3),2);
            f_psi_non_osc_final = f;
        end
    end
    
    psi_non_osc_final = mean(psi_non_osc,2);
end
