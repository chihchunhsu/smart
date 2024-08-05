import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from scipy.interpolate import interp1d
try:
    from scipy.integrate import trapezoid as trapz
except:
    from scipy.integrate import trapz

def integralResample(xh, yh, xl, nsamp=100, **kwargs):

    '''
    Function from SPLAT. See: https://github.com/aburgasser/splat

    :Purpose: A 1D integral smoothing and resampling function that attempts to preserve total flux. Usese
    scipy.interpolate.interp1d and scipy.integrate.trapz to perform piece-wise integration

    Required Inputs:

    :param xh: x-axis values for "high resolution" data
    :param yh: y-axis values for "high resolution" data
    :param xl: x-axis values for resulting "low resolution" data, must be contained within high resolution and have fewer values

    Optional Inputs:

    :param nsamp: Number of samples for stepwise integration

    Output:

    y-axis values for resulting "low resolution" data

    :Example:
    >>> # a coarse way of downsampling spectrum
    >>> import splat, numpy
    >>> sp = splat.Spectrum(file='high_resolution_spectrum.fits')
    >>> w_low = numpy.linspace(numpy.min(sp.wave.value),numpy.max(sp.wave.value),len(sp.wave.value)/10.)
    >>> f_low = splat.integralResample(sp.wave.value,sp.flux.value,w_low)
    >>> n_low = splat.integralResample(sp.wave.value,sp.noise.value,w_low)
    >>> sp.wave = w_low*sp.wave.unit
    >>> sp.flux = f_low*sp.flux.unit
    >>> sp.noise = n_low*sp.noise.unit
    '''

    """
    note: reducing nsamp from 100 -> 10 does not reduce time
    test how much precision is lost: try calculating integrated flux using
    splat method vs. bovy method
    """

    method = kwargs.get('method', 'fast')

    # check inputs
    #print('1', xl)
    #print('2', xh)
    #print('3', xl[0], xl[-1])
    #print('4', xh[0], xh[-1])
    if xl[0] < xh[0] or xl[-1] > xh[-1]: 
        raise ValueError('\nLow resolution x range {} to {} must be within high resolution x range {} to {}'.format(xl[0],xl[-1],xh[0],xh[-1]))
    if len(xl) > len(xh): 
        raise ValueError('\nTarget x-axis must be lower resolution than original x-axis')

    # Use flux preserving resample method
    if method == 'splat':
        # set up samples
        xs = [np.max([xh[0],xl[0]-0.5*(xl[1]-xl[0])])]
        for i in range(len(xl)-1): 
            xs.append(xl[i]+0.5*(xl[i+1]-xl[i]))
        xs.append(np.min([xl[-1]+0.5*(xl[-1]-xl[-2]),xh[-1]]))

        f = interp1d(xh,yh)

        # integral loop
        ys = []
        for i in range(len(xl)):
            dx = np.linspace(xs[i],xs[i+1],nsamp)
            ys.append(trapz(f(dx),x=dx)/trapz(np.ones(nsamp),x=dx))

    # Fast resample method
    elif method == 'fast':
        baseline = np.polynomial.Polynomial.fit(xh, yh, 4)
        ip       = interpolate.InterpolatedUnivariateSpline(xh, yh/baseline(xh), k=3)
        ys       = baseline(xl)*ip(xl)

    return ys




