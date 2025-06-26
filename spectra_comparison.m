function bf10 = spectra_comparison(spectra_1,spectra_2, tail)
    % This function performs a paired t-test using the values of each
    % frequency component from two populations of spectra. It includes code
    % from Bart Krekelberg (2025). bayesFactor
    % (https://github.com/klabhub/bayesFactor), GitHub. Retrieved March 3,
    % 2025.
    
    %% Input arguments:
    % spectra_1 and spectra_2: m x n matrices, being m each freqeuncy
    % component and n each spectrum.
    % tail: 'both','right', or 'left' for two or one-tailed tests [both].
    % Note that 'right' means spectra_1 > spectra_2.
    
    %% Output arguments:
    % bf10: Bayes factor
    
    for j = 1:size(spectra_1,1)
        [bf10(j),~] = ttest_bf(spectra_1(j,:),spectra_2(j,:),'tail',tail);
    end
end
