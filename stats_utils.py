# File: stats_utils.py
import numpy as np
from scipy import stats
from scipy.integrate import quad

def ttest_bf(X=None, *args, **kwargs):
    """
    TTEST Bayes Factors for one-sample and paired t-tests.
    bf10, pValue = ttest_bf(X, **opts)  - one sample
    bf10, pValue = ttest_bf(X, M, **opts)   - one sample, non-zero mean
    bf10, pValue = ttest_bf(X, Y, **opts)   - paired samples

    bf10, pValue = ttest_bf(T=T, N=10)   - calculate BF based
    on regular ttest output

    INPUT:
    X : array-like, single sample observations (column vector)
    Y : array-like, paired observations (column vector) or a scalar mean to compare X to. [default=0]

    Optional keyword arguments:
    tail  : 'both', 'right', or 'left' for two- or one-tailed tests [default: 'both']
    scale : scale of the Cauchy prior on the effect size [default: np.sqrt(2)/2]
    T     : t-value from a standard t-test
    N     : number of samples used for the t-test

    OUTPUT:
    bf10   : Bayes Factor for the hypothesis that the mean is different from zero
    pValue : p value of the frequentist hypothesis test
    """
    # Parse inputs
    tail = kwargs.get('tail', 'both').lower()
    scale = kwargs.get('scale', np.sqrt(2) / 2)
    T = kwargs.get('T', None)
    N = kwargs.get('N', None)

    # Determine data inputs
    if X is not None:
        # Data-based frequentist test
        if T is None:
            # X and optional Y provided
            if len(args) > 0:
                Y = args[0]
                # stats.ttest_rel for paired, stats.ttest_1samp for one-sample
                if np.isscalar(Y):
                    # one-sample test against scalar
                    tstat, pValue = stats.ttest_1samp(X, popmean=Y, alternative=None if tail=='both' else tail)
                    df = len(X) - 1
                    N = len(X)
                else:
                    # paired test
                    tstat, pValue = stats.ttest_rel(X, Y, alternative=None if tail=='both' else tail)
                    df = len(X) - 1
                    N = len(X)
            else:
                # one-sample test against zero
                Y = 0
                tstat, pValue = stats.ttest_1samp(X, popmean=0, alternative=None if tail=='both' else tail)
                df = len(X) - 1
                N = len(X)
            T = tstat
        else:
            # User specified T; trust kwargs N exists
            if N is None:
                raise ValueError('N must be specified when calling ttest_bf with a T')
            if np.iterable(N) and len(N) == 2:
                df = sum(N) - 2
                N = np.prod(N) / sum(N)
            else:
                df = N - 1
            # Compute pValue from T and df
            if tail == 'both':
                pValue = 2 * stats.t.sf(abs(T), df)
            elif tail == 'right':
                pValue = stats.t.sf(T, df)
            elif tail == 'left':
                pValue = stats.t.cdf(T, df)
    else:
        # No X provided, must use T and N directly
        if T is None or N is None:
            raise ValueError('Must specify X or both T and N')
        if np.iterable(N) and len(N) == 2:
            df = sum(N) - 2
            N = np.prod(N) / sum(N)
        else:
            df = N - 1
        if tail == 'both':
            pValue = 2 * stats.t.sf(abs(T), df)
        elif tail == 'right':
            pValue = stats.t.sf(T, df)
        elif tail == 'left':
            pValue = stats.t.cdf(T, df)

    # Compute Bayes Factor using Rouder et al.
    # numerator
    numerator = (1 + T**2 / df) ** (-(df + 1) / 2)

    # integrand function
    def integrand(g):
        return ((1 + N * g * scale**2) ** -0.5) * \
               ((1 + T**2 / ((1 + N * g * scale**2) * df)) ** (-(df + 1) / 2)) * \
               (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1.0 / (2 * g))

    # integrate from 0 to infinity
    integral_val, _ = quad(integrand, 0, np.inf)

    bf01 = numerator / integral_val
    bf10 = 1.0 / bf01

    # adjust for one-sided
    if tail in ['left', 'right']:
        if pValue == 1:
            pValue = 0.9999999999999
        bf10 = 2 * (1 - pValue) * bf10

    return bf10, pValue

def spectra_comparison(spectra_1, spectra_2, tail='both'):
    """
    This function performs a paired t-test using the values of each
    frequency component from two populations of spectra. It includes code
    from Bart Krekelberg (2025). bayesFactor
    (https://github.com/klabhub/bayesFactor), GitHub. Retrieved March 3,
    2025.

    INPUT ARGUMENTS:
    spectra_1, spectra_2 : array-like, shape (m, n)
        m frequency components and n spectra per population
    tail : {'both', 'right', 'left'}
        two- or one-tailed tests [default: 'both'].
        Note that 'right' means spectra_1 > spectra_2.

    OUTPUT ARGUMENTS:
    bf10 : ndarray, shape (m,)
        Bayes factor for each frequency component
    """
    m = spectra_1.shape[0]
    bf10 = np.empty(m)

    for j in range(m):
        bf10[j], _ = ttest_bf(spectra_1[j, :], spectra_2[j, :], tail=tail)

    return bf10
