#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.special import wofz
import smart
import copy


def continuum(data, mdl, deg=10, prop=False, tell=False):
    
    """
    This function returns a continuum corrected model.
    Ported from apogee_tools.
    
    Parameters
    ----------
    data    :   Spectrum object
                the data for the continuum fit
    mdl     :   Model object
                the stellar atmosphere model
    deg     :   int 
                the degree of the fitting polynomial. 
                The default vaule is 10.
    
    Returns
    -------
    mdl     :   an instance of the Model object
                A continuum corrected model by the fitted polynomial
    """
    #print('data wave:', type(data.wave[0]), data.wave[-1], data.wave.shape)
    #print('model wave:', type(mdl.wave), mdl.wave[-1])
    #print(data.wave)
    #print(mdl.wave)
    if data.instrument in ['nirspec', 'hires', 'igrins', 'kpic', 'fire']:
        #print('model',mdl.wave)
        mdl_range      = np.where((mdl.wave >= data.wave[0]) & (mdl.wave <= data.wave[-1]))
        mdl_wave       = mdl.wave[mdl_range]
        mdl_flux       = mdl.flux[mdl_range]
        #sys.exit()
    #elif data.instrument in ['igrins']:
    #    # avoid the telluric CH4 region for order 6
    #    mdl_range      = np.where( ((mdl.wave >= data.wave[0]) & (mdl.wave <= data.wave[1350])) | (mdl.wave >= data.wave[1650]) & (mdl.wave <= data.wave[-1])  )
    #    mdl_wave       = mdl.wave[mdl_range]
    #    mdl_flux       = mdl.flux[mdl_range]

    elif data.instrument in ['apogee']:
        ## the index for apogee is reversed
        #mdl_range      = np.where((mdl.wave >= data.wave[-1]) & (mdl.wave <= data.wave[0]))
        mdl_wave       = mdl.wave
        mdl_flux       = mdl.flux

    if data.instrument == 'igrins':
        # avoid the telluric CH4 region for order 6
        wave = np.concatenate((data.wave[70: 1350], data.wave[1650: -10]), axis=None)
        flux = np.concatenate((data.flux[70: 1350], data.flux[1650: -10]), axis=None)

        mdl_int         = np.interp(wave, mdl_wave, mdl_flux)
        mdldiv          = flux/mdl_int

    else:

        mdl_int         = np.interp(data.wave, mdl_wave, mdl_flux)
        mdldiv          = data.flux/mdl_int

    ## find mean and stdev of mdldiv
    mean_mdldiv     = np.nanmean(mdldiv)
    std_mdldiv      = np.nanstd(mdldiv)
    
    ## replace outliers with average value for nirspec
    if data.instrument in ['nirspec', 'hires', 'kpic','fire']:
        mdldiv[mdldiv  <= mean_mdldiv - 2 * std_mdldiv] = mean_mdldiv
        mdldiv[mdldiv  >= mean_mdldiv + 2 * std_mdldiv] = mean_mdldiv
        #try:
        pcont           = np.polyfit(data.wave, mdldiv, deg, w=1/data.noise**2)
    #except:
        #    ## if the length of the data flux and noise are not the same
        #    pcont           = np.polyfit(data.wave, mdldiv, deg)
    

    elif data.instrument in ['igrins']:
        #mdldiv[mdldiv  <= mean_mdldiv - 2 * std_mdldiv] = mean_mdldiv
        #mdldiv[mdldiv  >= mean_mdldiv + 2 * std_mdldiv] = mean_mdldiv
        #try:
        #   pcont           = np.polyfit(wave, mdldiv, deg, w=1/data.noise**2)
        #except:
        #    ## if the length of the data flux and noise are not the same
        pcont           = np.polyfit(wave, mdldiv, deg)

    ## outlier rejection for apogee
    elif data.instrument == 'apogee':
        select_poly_fit = np.where(np.absolute(mdldiv - mean_mdldiv) <= 2 * std_mdldiv)
        mdldiv          = mdldiv[select_poly_fit]
        data_wave_fit   = data.wave[select_poly_fit]
        pcont           = np.polyfit(data_wave_fit, mdldiv, deg)

    mdl.flux       *= np.polyval(pcont, mdl.wave)

    if data.instrument == 'apogee':
        constA    = (np.std(data.flux[select_poly_fit])/np.std(mdl.flux[select_poly_fit]))
        mdl.flux *= constA
        constB    = np.median(mdl.flux[select_poly_fit]) - np.median(data.flux[select_poly_fit])
        mdl.flux -= constB

    if prop:
        if data.instrument == 'apogee':
            return mdl, np.polyval(pcont, mdl.wave), constA, constB
        else:
            return mdl, np.polyval(pcont, mdl.wave)
    elif tell:
        return mdl, pcont
    else:
        return mdl

#def _continuumFit(x, c0, c1, c2, c3):
#    #return c0 + c1*x + c2*x**2 + c3*x**3
#    return np.poly1d([c3, c2, c1, c0])(x)

#def _continuumFit(x, c1, c0):
#  return np.poly1d([c1, c0])(x)

def _continuumFit(x, c2, c1, c0):
   return np.poly1d([c2, c1, c0])(x)

def linear_fit(x,a,b):
    return a*x + b

def gaus_absorption_only(x,x0,sigma,a,b):
    return (1 - b*np.e**(-(x-x0)**2/(2*sigma**2)))*a

def gaus_absorption_spec(x, x0, sigma, scale, a, b, c, d):
    """
    This function is to fit continuum of the spectra 
    with one absorption feature.
    
    """
    gaussian_absorption = 1 - scale*np.e**(-(x-x0)**2/(2*sigma**2))
    return gaussian_absorption * (a*x**2 + b*x + c) + d

def voigt_profile(x, x0, amp, gamma, scale, a, b, c, d):
    """
    Return a spectral line absorption with a second order 
    polynomial and the Voigt line shape absorption at x0 
    with Lorentzian component HWHM gamma and Gaussian component
    HWHM alpha (the latter is absorbed in the amp parameter.

    """
    #sigma = alpha / np.sqrt(2 * np.log(2))

    voigt_absorption = (1 - scale * np.real(wofz((x-x0 + 1j*gamma)*amp)))
    return voigt_absorption * (a*x**2 + b*x + c) + d

def voigt_profile_emission(x, x0, amp, gamma, scale, a, b, c, d):
    """
    Return a spectral line emission with a second order 
    polynomial and the Voigt line shape absorption at x0 
    with Lorentzian component HWHM gamma and Gaussian component
    HWHM alpha (the latter is absorbed in the amp parameter.

    """
    #sigma = alpha / np.sqrt(2 * np.log(2))

    voigt_profile = (1 + scale * np.real(wofz((x - x0 + 1j*gamma)*amp)))
    
    return voigt_profile * (a*x**2 + b*x + c) + d

def voigt_profile_emission_double(x, x0, sigma, gamma, scale, x1, sigma1, gamma1, scale1, a, b, c, d):
    """
    Return a spectral line emission with a second order 
    polynomial and two Voigt line shape absorption at x0 and x2
    with Lorentzian component HWHM gamma and Gaussian component
    HWHM alpha (the latter is absorbed in the amp parameter.

    """
    #sigma  = alpha / np.sqrt(2 * np.log(2))
    #sigma1 = alpha / np.sqrt(2 * np.log(2))

    voigt_profile = (1 + scale * np.real(wofz((x - x0 + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) \
        + scale1 * np.real(wofz((x - x1 + 1j*gamma1)/sigma1/np.sqrt(2))) / sigma1 /np.sqrt(2*np.pi))

    return voigt_profile * (a*x**2 + b*x + c) + d

def sineFit(wavelength,frequency,amplitude,phase,offset):
    """
    A sine fit function of wavelength, frequency,
    amplitude, and phase.
    """
    return amplitude * np.sin(frequency * (wavelength - phase)) + offset

def fringeTelluric(data):
    """
    Model the fringe pattern for telluric data.
    
    Note: The input data should be continuum corrected 
    before using this function.
    """
    lsf       = smart.getLSF(data)
    alpha     = smart.getAlpha(data, lsf)
    tell_mdl2 = smart.convolveTelluric(lsf=lsf,
        telluric_data=data,alpha=alpha)
    
    pgram_x  = np.array(data.wave,float)[10:-10]
    pgram_y  = np.array(data.flux-tell_mdl2.flux,float)[10:-10]
    offset   = np.mean(pgram_y)
    pgram_y -= offset
    mask     = np.where(np.absolute(pgram_y)-1.*np.std(pgram_y)>0)
    pgram_x  = np.delete(pgram_x,mask)
    pgram_y  = np.delete(pgram_y,mask)
    pgram_x  = np.array(pgram_x,float)
    pgram_y  = np.array(pgram_y,float)

    f        = np.lismartace(0.01,10,100000)

    ## Lomb Scargle Periodogram
    pgram    = signal.lombscargle(pgram_x, pgram_y, f)

    freq     = f[np.argmax(pgram)]

    ## initial guess for the sine fit
    amp0     = np.absolute(np.std(pgram_y))
    p0       = [freq, amp0, 0, 0]
    popt, pcov = curve_fit(sineFit, pgram_x, pgram_y, p0=p0,
        maxfev=100000)

    #data.wave = pgram_x
    #data.flux = np.delete(data.flux[10:-10],
    #    mask)-(sineFit(pgram_x,*popt)-popt[-1])
    #data.noise = np.delete(data.noise[10:-10],mask)
    data.flux -= (sineFit(data.wave,*popt)-popt[-1])

    return data, sineFit(data.wave,*popt)-popt[-1]

def continuumTelluric(data, model=None):
    """
    Return a continnum telluric standard data.
    Default: return a telluric flux of mean 1.

    Parameters
    ----------
    data:  spectrum object
           The input telluric data to be continuum
           corrected

    model: (optional) model object
           The telluric model to obtain the mean flux
           Instead of 1 as in default, it returns a constant
           shift by the difference between the mean flux of 
           the telluric data and that of the telluric model

    Returns
    -------
    data: spectrum object
          continuum corrected telluric data

    Examples
    --------
    >>> import smart
    >>> smart.continuumTelluric(data)

    >>> smart.continuumTelluric(data,model)

    """
    if model is None:
        wavelow  = data.wave[0] - 20
        wavehigh = data.wave[-1] + 20
        model    = smart.getTelluric(wavelow,wavehigh)

    if not data.apply_sigma_mask:
        data2 = copy.deepcopy(data)
        data.maskBySigmas(sigma=1.5)
    else:
        data2 = copy.deepcopy(data)
        data.wave  = data.wave[10:-30]
        data.flux  = data.flux[10:-30]
        data.noise = data.noise[10:-30]

    #    plt.plot(data.wave,data.flux, alpha=0.5)
    #    plt.plot(data2.wave,data2.flux, alpha=0.5)
    #    plt.show()
    #    plt.close()

    if data.order == 35:
        # O35 has a voigt absorption profile
        popt, pcov = curve_fit(voigt_profile,data.wave[20:-20],
            data.flux[20:-20],
            p0=[21660,2000,0.1,0.1,0.01,0.1,10000,1000],
            maxfev=10000)
        #plt.plot(data.wave,data.flux,'k-',alpha=0.5)
        #plt.plot(data.wave,voigt_profile(data.wave,*popt),'r-',alpha=0.5)
        #plt.show()
        #plt.close()
        const = np.mean(data.flux/voigt_profile(data.wave, *popt))\
        -np.mean(model.flux)
        data.flux  = data.flux/voigt_profile(data.wave, *popt) - const
        data.noise = data.noise/voigt_profile(data.wave, *popt)
        #if not data.applymask:
        data2.flux  = data2.flux/voigt_profile(data2.wave, *popt) - const
        data2.noise = data2.noise/voigt_profile(data2.wave, *popt)
        data        = data2

    elif data.order == 38 or data.order == 30:
        # O38 has rich absorption features
        def fit_continuum_O38(x,a,b,**kwargs):
            flux = kwargs.get('flux',data.flux)
            linear = a*x + b
            return flux/linear

        model2      = copy.deepcopy(model)
        model2.flux = smart.broaden(wave=model2.wave, 
            flux=model2.flux, vbroad=4.8, 
            rotate=False, gaussian=True)
        model2.flux = np.array(smart.integralResample(xh=model2.wave, 
            yh=model2.flux, xl=data.wave))
        model2.wave = data.wave
        
        popt, pcov  = curve_fit(fit_continuum_O38,data.wave,
            model2.flux, p0=[ 8.54253062e+00  , -166000])
        #const = np.mean(data.flux/linear_fit(data.wave, *popt))-np.mean(model.flux)
        #data.flux = data.flux/linear_fit(data.wave, *popt) - const
        data.flux   = data.flux/linear_fit(data.wave, *popt)
        data.noise  = data.noise/linear_fit(data.wave, *popt)
        #if not data.applymask:
        data2.flux  /= linear_fit(data2.wave, *popt)
        data2.noise /= linear_fit(data2.wave, *popt)
        data         = data2

    elif data.order == 55:
        popt, pcov  = curve_fit(_continuumFit, data.wave, data.flux)
        #if data.applymask:
        #    data.flux   = data.flux/_continuumFit(data.wave, *popt)
        #    data.noise  = data.noise/_continuumFit(data.wave, *popt)
        #
        #    factor      = np.max(data.flux)
        #    data.flux  /= factor
        #    data.noise /= factor
        #    data.flux  /= 0.93
        #    data.noise /= 0.93

        #elif not data.applymask:
        data2.flux   = data2.flux/_continuumFit(data2.wave, *popt)
        data2.noise  = data2.noise/_continuumFit(data2.wave, *popt)

        factor       = np.max(data2.flux)
        data2.flux   = data2.flux/factor
        data2.noise  = data2.noise/factor

        data2.flux  /= 0.93
        data2.noise /= 0.93

        data         = data2

    elif data.order == 56:
        # select the highest points to fit a polynomial
        x1 = np.max(data.flux[0:100])
        x2 = np.max(data.flux[100:150])
        x3 = np.max(data.flux[200:300])
        x4 = np.max(data.flux[300:400])
        x5 = np.max(data.flux[600:700])
        x6 = np.max(data.flux[700:800])

        a = [float(data.wave[np.where(data.flux == x1)]),
        float(data.wave[np.where(data.flux == x2)]),
        float(data.wave[np.where(data.flux == x3)]),
        float(data.wave[np.where(data.flux == x4)]),
        float(data.wave[np.where(data.flux == x5)]),
        float(data.wave[np.where(data.flux == x6)])]
        
        b = [x1, x2, x3, x4, x5, x6]

        popt, pcov = curve_fit(_continuumFit, a, b)

        data.flux = data.flux/_continuumFit(data.wave,*popt)*0.85
        data.noise = data.noise/_continuumFit(data.wave,*popt)*0.85
        #if not data.applymask:
        data2.flux  = data2.flux/_continuumFit(data2.wave,*popt)*0.85
        data2.noise = data2.noise/_continuumFit(data2.wave,*popt)*0.85
        data        = data2

    elif data.order == 59:
        #wave0 = int(data.wave[np.where(data.flux==np.min(data.flux))])
        popt, pcov = curve_fit(voigt_profile,
            data.wave, data.flux,
            p0=[12820,2000,0.1,0.1,0.01,0.1,10000,1000],maxfev=100000)
            #p0=[wave0,2000,0.1,0.1,0.01,0.1,10000,1000],
            #maxfev=10000)        
        data.flux  /= voigt_profile(data.wave, *popt)
        data.noise /= voigt_profile(data.wave, *popt)
        #if not data.applymask:
        data2.flux  /= voigt_profile(data2.wave, *popt)
        data2.noise /= voigt_profile(data2.wave, *popt)
        data         = data2
        #plt.plot(data.wave,data.flux)
        #plt.show()
        #plt.close()

    ## this is not true in general!!
    elif data.order == 65:
        # O65 is best mateched by a gaussian absorption feature
        popt, pcov  = curve_fit(gaus_absorption_only,
            data.wave,data.flux, p0=[11660,50,2000,2000],maxfev=100000)
        const       = np.mean(data.flux/gaus_absorption_only(data.wave, 
            *popt)) - np.mean(model.flux)
        data.flux   = data.flux/gaus_absorption_only(data.wave, *popt) - const
        data.noise /= gaus_absorption_only(data.wave, *popt)
        #if not data.applymask:
        data2.flux   = data2.flux/gaus_absorption_only(data2.wave, *popt) - const
        data2.noise /= gaus_absorption_only(data2.wave, *popt)
        data         = data2

    else:
        # this second order polynomial continnum correction 
        # works for the O33, O34, O36, and O37
        idx = np.isfinite(data.wave) & np.isfinite(data.flux)
        popt, pcov = curve_fit(_continuumFit, data.wave[idx], data.flux[idx])
        const      = np.mean(data.flux/_continuumFit(data.wave, *popt)) - np.mean(model.flux)
        if data.order == 57: const = 0
        data.flux  = data.flux/_continuumFit(data.wave, *popt) - const
        data.noise = data.noise/_continuumFit(data.wave, *popt)
        #if not data.applymask:
        data2.flux  = data2.flux/_continuumFit(data2.wave, *popt) - const
        data2.noise = data2.noise/_continuumFit(data2.wave, *popt)
        data        = data2

    return data
