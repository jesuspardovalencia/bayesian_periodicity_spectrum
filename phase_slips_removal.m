function f_inst_final = phase_slips_removal(f_inst, Fs, min_window, max_window, n_win)
    % Based on "Fluctuations in Oscillation Frequency Control Spike Timing
    % and Coordinate Neural Networks", published on The Journal of
    % Neuroscience (https://doi.org/10.1523/JNEUROSCI.0261-14.2014).
    
    %% Input arguments:
    % f_inst: Instantaneous frequency matrix [time x frequencies].
    % Fs: Sampling frequency (Hz).
    % min_window: Minimum window length (in ms) for median filtering.
    % max_window: Maximum window length (in ms) for median filtering.
    % n_win: Number of windows for median filtering.

    %% Output arguments:
    % f_inst_final: Corrected matrix [time x frequencies].
   
    windows = min_window:(max_window-min_window)/(n_win-1):max_window;
    for j = 1:1:length(windows)
        f_inst_filtered(:,:,j) = movmedian(f_inst, round(windows(j)*Fs/1000));
    end
    f_inst_final = median(f_inst_filtered,3);
end
