function f_inst_final = phase_jumps_correction(f_inst, f, Fs)
    % This function corrects jumps of +- 2 * pi in the instantaneous
    % frequency traces.
    
    %% Input arguments:
    % f_inst: Instantaneous frequency matrix [time x frequencies].
    % f: Frequencies array (Hz).
    % Fs: Sampling frequency (Hz).

    %% Output argument:
    % f_inst_final: Corrected matrix [time x frequencies].
    
    f_inst_final = zeros(size(f_inst));
    threshold = (Fs/2);
    for z = 1:1:size(f_inst,2)
        f_inst_t = f_inst(:,z);
        jumps = [];
        jumps = find(abs(diff(f_inst_t)) > threshold);
        for j = 1:length(jumps)
            idx = jumps(j) + 1:length(f_inst_t);
            if f_inst_t(jumps(j) + 1) > f_inst_t(jumps(j))
                f_inst_t(idx) = f_inst_t(idx) - (2 * threshold);
            else
                f_inst_t(idx) = f_inst_t(idx) + (2 * threshold);
            end
            if mean(f_inst_t(idx)) < f(z) - (2 * threshold)
                f_inst_t(idx) = f_inst_t(idx) + (2 * threshold);
            elseif mean(f_inst_t(idx)) > f(z) + (2 * threshold)
                f_inst_t(idx) = f_inst_t(idx) - (2 * threshold);
            end
        end
        idx_1 = f_inst_t > f(z) + (threshold);
        f_inst_t(idx_1) = f_inst_t(idx_1) - (2 * threshold);
        idx_1 = f_inst_t < f(z) - (threshold);
        f_inst_t(idx_1) = f_inst_t(idx_1) + (2 * threshold); 
        f_inst_final(:,z) = f_inst_t;
    end
end
