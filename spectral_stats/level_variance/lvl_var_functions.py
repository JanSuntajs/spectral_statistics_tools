import numpy as np
import numba as nb


@nb.njit('float64[:](float64, float64[:])', fastmath=True, nogil=True)
def sigma_num_variance(L, E):
    """
    A function that calculates number variance for
    an energy spectrum.

    Definition of sigma:

        sigma**2(L)=<N(L)**2> - <N(L)>**2

    where L is the width of the energy interval in the spectrum
    and N is the number of energy levels inside the interval L.
    < ... > denotes a moving window average over different levels
    in the spectrum.

    INPUT:

    L - energy interval width
    E - energies, a particular energy spectrum

    OUTPUT:

    A vector containing three quantities:
    <N(L)>
    sigma**2(L)
    variance of sigma**2

    """

    result = np.zeros(shape=3, dtype=np.float64)

    Ave1 = 0.0
    Ave2 = 0.0
    Ave3 = 0.0
    Ave4 = 0.0

    j = 1
    k = 1

    x = E[0]

    ndata = E.shape[0]

    # Count the number of values in the interval.
    # If L is large, make sure that the loop
    # does not iterate over the limit.
    while ((E[k] < x + L)):

        if k == ndata - 1:
            break
        else:
            k += 1
    # Perform the actual calculation -
    # Move the energy window over the energy interval
    # and sum the contributions to Ave1, Ave2, Ave3, Ave4
    # during the process.
    # while (x <= E[ndata-1]):
    while ((k <= ndata - 1) and (x <= E[ndata - 1])):

        d1 = E[j] - x
        d2 = E[k] - (x + L)
        cn = 1. * (k - j)

        if d1 < d2:
            x = E[j]
            s = d1
            j += 1
        else:
            x = E[k] - L
            s = d2
            k += 1

        Ave1 = Ave1 + s * cn
        Ave2 = Ave2 + s * cn ** 2
        Ave3 = Ave3 + s * cn ** 3
        Ave4 = Ave4 + s * cn ** 4

    s = x - E[0]
    # s = E[ndata -1] - E[0]

    # Ave1 = Ave1 / s
    Ave1 = Ave1 / s
    Ave2 = Ave2 / s
    Ave3 = Ave3 / s
    Ave4 = Ave4 / s

    AveNum = Ave1  # AveNum
    AveSig = Ave2 - AveNum ** 2  # AveSig
    VarSig = 1. * Ave4 - 4. * Ave3 * AveNum + 8. * Ave2 * AveNum ** 2 - \
        4. * AveNum ** 4 - Ave2 ** 2

    result[0] = AveNum
    result[1] = AveSig
    result[2] = VarSig

    return result


@nb.njit('float64[:,:](float64[:], float64[:])', fastmath=True,
         parallel=True, nogil=True)
def sigma_loop(L, E):
    """
    A function that calculates sigma**2(L) dependence
    by calling the sigma_num_variance function
    for different L values.

    INPUT:

    L - an array of L values
    E - an array of energy values, a particular energy spectrum

    OUTPUT:
    (arrays) for <N(L)>, sigma**2(L) and variance of sigma**2.

    """

    # ndata = E.shape[0]
    nL = L.shape[0]

    NumVar = np.zeros(shape=(3, nL), dtype=np.float64)
    res_size = 3
    result = np.zeros(shape=res_size, dtype=np.float64)

    for i in nb.prange(nL):

        result = sigma_num_variance(L[i], E)

        for j in range(res_size):
            NumVar[j, i] = result[j]

    return NumVar


@nb.jit('float64[:,:](float64[:], float64[:,:], b1)',
        fastmath=True, forceobj=True)
def sigma_averaged(L, E, deviation=False):
    """
    A function that averages different sigma loop
    calculations obtained for different disorder
    realizations.

    INPUT:

    L - a list of L values
    E - a list of energy spectra for different
        disorder realizations
    deviation - boolean, whether deviations
                are also calculated

    If deviation == False:
        The output shape is the same as in the
        sigma_loop function's case.
    else:
        Also the errors for the quantities are returned

    """
    nsamples = E.shape[0]
    nL = L.shape[0]

    if not deviation:
        results = np.zeros(shape=(3, nL), dtype=np.float64)

        for i in range(nsamples):

            results += sigma_loop(L, E[i])

        return results * 1. / nsamples
    else:
        results = np.zeros(shape=(6, nL), dtype=np.float64)

        partial = np.zeros(shape=(nsamples, 3, nL), dtype=np.float64)

        for i in range(nsamples):

            partial[i] = sigma_loop(L, E[i])

        results[:3] = np.mean(partial, axis=0)
        results[3:] = np.std(partial, axis=0)

        return results
