import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import smart
import emcee
import corner
import copy
import time
import os
import sys

def makeModel(teff, logg=5, metal=0, vsini=1,rv=0, tell_alpha=1.0, airmass=1.0, pwv=0.5, wave_offset=0, flux_offset=0,**kwargs):
    """
    Return a forward model.

    Parameters
    ----------
    teff   : effective temperature
    
    data   : an input science data used for continuum correction

    Optional Parameters
    -------------------
    

    Returns
    -------
    model: a synthesized model
    """

    # read in the parameters
    order      = kwargs.get('order', '33')
    modelset   = kwargs.get('modelset', 'btsettl08')
    instrument = kwargs.get('instrument', 'nirspec')
    veiling    = kwargs.get('veiling', 0)    # flux veiling parameter
    lsf        = kwargs.get('lsf', 4.5)   # instrumental LSF
    if instrument == 'apogee':
        try:
            import apogee_tools as ap
        except ImportError:
            print('Need to install the package "apogee_tools" (https://github.com/jbirky/apogee_tools) \n')
        xlsf       = kwargs.get('xlsf', np.linspace(-7.,7.,43))   # APOGEE instrumental LSF sampling
        wave_off1  = kwargs.get('wave_off1') # wavelength offset for chip a
        wave_off2  = kwargs.get('wave_off2') # wavelength offset for chip b
        wave_off3  = kwargs.get('wave_off3') # wavelength offset for chip c
        c0_1       = kwargs.get('c0_1')      # constant flux offset for chip a
        c0_2       = kwargs.get('c0_2')      # linear flux offset for chip a
        c1_1       = kwargs.get('c1_1')      # constant flux offset for chip b
        c1_2       = kwargs.get('c1_2')      # linear flux offset for chip b
        c2_1       = kwargs.get('c2_1')      # constant flux offset for chip c
        c2_2       = kwargs.get('c2_2')      # linear flux offset for chip c

    tell       = kwargs.get('tell', True) # apply telluric
    #tell_alpha = kwargs.get('tell_alpha', 1.0) # Telluric alpha power
    binary     = kwargs.get('binary', False) # make a binary model

    # assume the secondary has the same metallicity
    if binary:
        teff2       = kwargs.get('teff2')
        logg2       = kwargs.get('logg2')
        rv2         = kwargs.get('rv2')
        vsini2      = kwargs.get('vsini2')
        flux_scale = kwargs.get('flux_scale', 0.8)

    data       = kwargs.get('data', None) # for continuum correction and resampling

    output_stellar_model = kwargs.get('output_stellar_model', False)
    
    if data is not None and instrument == 'nirspec':
        order = data.order
        # read in a model
        #print('teff ',teff,'logg ',logg, 'z', z, 'order', order, 'modelset', modelset)
        #print('teff ',type(teff),'logg ',type(logg), 'z', type(z), 'order', type(order), 'modelset', type(modelset))
        model    = smart.Model(teff=teff, logg=logg, metal=metal, order=str(order), modelset=modelset, instrument=instrument)

    #elif data is not None and instrument == 'apogee':
    elif instrument == 'apogee':
        model    = smart.Model(teff=teff, logg=logg, metal=metal, modelset=modelset, instrument=instrument)
        # Dirty fix here
        model.wave = model.wave[np.where(model.flux != 0)]
        model.flux = model.flux[np.where(model.flux != 0)]

        # apply vmicro
        vmicro = 2.478 - 0.325*logg
        model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vmicro, rotate=False, gaussian=True)
    
    elif data is None and instrument == 'nirspec':
        model    = smart.Model(teff=teff, logg=logg, metal=metal, order=str(order), modelset=modelset, instrument=instrument)

    else:
        model    = smart.Model(teff=teff, logg=logg, metal=metal, order=str(order), modelset=modelset, instrument=instrument)

    
    # wavelength offset
    #model.wave += wave_offset

    # apply vsini
    model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True, gaussian=False)
    
    # apply rv (including the barycentric correction)
    model.wave = rvShift(model.wave, rv=rv)
    
    # flux veiling
    model.flux += veiling

    ## if binary is True: make a binary model
    if binary:
        model2      = smart.Model(teff=teff2, logg=logg2, metal=metal, order=str(order), modelset=modelset, instrument=instrument)
        # apply vsini
        model2.flux = smart.broaden(wave=model2.wave, flux=model2.flux, vbroad=vsini2, rotate=True, gaussian=False)
        # apply rv (including the barycentric correction)
        model2.wave = rvShift(model2.wave, rv=rv2)
        # linearly interpolate the model2 onto the model1 grid
        fit = interp1d(model2.wave, model2.flux)

        select_wavelength = np.where( (model.wave < model2.wave[-1]) & (model.wave > model2.wave[0]) )
        model.flux = model.flux[select_wavelength]
        model.wave = model.wave[select_wavelength]

        # combine the models together and scale the secondary flux
        model.flux += flux_scale * fit(model.wave)

    if output_stellar_model:
        stellar_model = copy.deepcopy(model)
        if binary:
            model2.flux = flux_scale * fit(model.wave)

    # apply telluric
    if tell is True:
        if instrument.lower() == 'apf':
            altitude = 1283
            model = smart.applyTelluric(model=model, tell_alpha=tell_alpha, altitude=altitude, pwv=pwv, instrument=instrument.lower())
        else:
            model = smart.applyTelluric(model=model, tell_alpha=tell_alpha, airmass=airmass, pwv=pwv,)

    # instrumental LSF
    if instrument == 'nirspec':
        model.flux = smart.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)
    elif instrument == 'apogee':
        model.flux = ap.apogee_hack.spec.lsf.convolve(model.wave, model.flux, lsf=lsf, xlsf=xlsf).flatten()
        model.wave = ap.apogee_hack.spec.lsf.apStarWavegrid()
        # Remove the NANs
        model.wave = model.wave[~np.isnan(model.flux)]
        model.flux = model.flux[~np.isnan(model.flux)]

    if output_stellar_model:
        stellar_model.flux = smart.broaden(wave=stellar_model.wave, flux=stellar_model.flux, vbroad=lsf, rotate=False, gaussian=True)
        if binary:
            model2.flux = smart.broaden(wave=model2.wave, flux=model2.flux, vbroad=lsf, rotate=False, gaussian=True)

    # add a fringe pattern to the model
    #model.flux *= (1+amp*np.sin(freq*(model.wave-phase)))

    # wavelength offset
    model.wave += wave_offset

    if output_stellar_model: 
        stellar_model.wave += wave_offset
        if binary:
            model2.wave = stellar_model.wave

    # integral resampling
    if data is not None:
        if instrument == 'nirspec':
            model.flux = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
            model.wave = data.wave

            if output_stellar_model:
                stellar_model.flux = np.array(smart.integralResample(xh=stellar_model.wave, yh=stellar_model.flux, xl=data.wave))
                stellar_model.wave = data.wave
                if binary:
                    model2.flux = np.array(smart.integralResample(xh=model2.wave, yh=model2.flux, xl=data.wave))
                    model2.wave = data.wave

        # contunuum correction
        if data.instrument == 'nirspec':
            niter = 5 # continuum iteration
            if output_stellar_model:
                model, cont_factor = smart.continuum(data=data, mdl=model, prop=True)
                for i in range(niter):
                    model, cont_factor2 = smart.continuum(data=data, mdl=model, prop=True)
                    cont_factor *= cont_factor2
                stellar_model.flux *= cont_factor
                if binary:
                    model2.flux *= cont_factor
            else:
                model = smart.continuum(data=data, mdl=model)
                for i in range(niter):
                    model = smart.continuum(data=data, mdl=model)
        elif data.instrument == 'apogee':
            ## set the order in the continuum fit
            deg         = 5
            ## because of the APOGEE bands, continuum is corrected from three pieces of the spectra
            data0       = copy.deepcopy(data)
            model0      = copy.deepcopy(model)

            # wavelength offset
            model0.wave += wave_off1

            range0      = np.where((data0.wave >= data.oriWave0[0][-1]) & (data0.wave <= data.oriWave0[0][0]))
            data0.wave  = data0.wave[range0]
            data0.flux  = data0.flux[range0]
            if data0.wave[0] > data0.wave[-1]:
                data0.wave = data0.wave[::-1]
                data0.flux = data0.flux[::-1]
            model0.flux = np.array(smart.integralResample(xh=model0.wave, yh=model0.flux, xl=data0.wave))
            model0.wave = data0.wave
            model0      = smart.continuum(data=data0, mdl=model0, deg=deg)
            # flux corrections
            model0.flux = (model0.flux + c0_1) * np.e**(-c0_2)

            data1       = copy.deepcopy(data)
            model1      = copy.deepcopy(model)

            # wavelength offset
            model1.wave += wave_off2

            range1      = np.where((data1.wave >= data.oriWave0[1][-1]) & (data1.wave <= data.oriWave0[1][0]))
            data1.wave  = data1.wave[range1]
            data1.flux  = data1.flux[range1]
            if data1.wave[0] > data1.wave[-1]:
                data1.wave = data1.wave[::-1]
                data1.flux = data1.flux[::-1]
            model1.flux = np.array(smart.integralResample(xh=model1.wave, yh=model1.flux, xl=data1.wave))
            model1.wave = data1.wave
            model1      = smart.continuum(data=data1, mdl=model1, deg=deg)

            # flux corrections
            model1.flux = (model1.flux + c1_1) * np.e**(-c1_2)

            data2       = copy.deepcopy(data)
            model2      = copy.deepcopy(model)

            # wavelength offset
            model2.wave += wave_off3

            range2      = np.where((data2.wave >= data.oriWave0[2][-1]) & (data2.wave <= data.oriWave0[2][0]))
            data2.wave  = data2.wave[range2]
            data2.flux  = data2.flux[range2]
            if data2.wave[0] > data2.wave[-1]:
                data2.wave = data2.wave[::-1]
                data2.flux = data2.flux[::-1]

            model2.flux = np.array(smart.integralResample(xh=model2.wave, yh=model2.flux, xl=data2.wave))
            model2.wave = data2.wave
            model2      = smart.continuum(data=data2, mdl=model2, deg=deg)
            # flux corrections
            model2.flux = (model2.flux + c2_1) * np.e**(-c2_2)

            ## scale the flux to be the same as the data
            #model0.flux *= (np.std(data0.flux)/np.std(model0.flux))
            #model0.flux -= np.median(model0.flux) - np.median(data0.flux)

            #model1.flux *= (np.std(data1.flux)/np.std(model1.flux))
            #model1.flux -= np.median(model1.flux) - np.median(data1.flux)

            #model2.flux *= (np.std(data2.flux)/np.std(model2.flux))
            #model2.flux -= np.median(model2.flux) - np.median(data2.flux)

            model.flux  = np.array( list(model2.flux) + list(model1.flux) + list(model0.flux) )
            model.wave  = np.array( list(model2.wave) + list(model1.wave) + list(model0.wave) )

    if instrument == 'nirspec':
        # flux offset
        model.flux += flux_offset
        if output_stellar_model: 
            stellar_model.flux += flux_offset
            if binary:
                model2.flux += flux_offset
    #model.flux **= (1 + flux_exponent_offset)

    if output_stellar_model:
        if not binary:
            return model, stellar_model
        else:
            return model, stellar_model, model2
    else:
        return model

def rvShift(wavelength, rv):
    """
    Perform the radial velocity correction.

    Parameters
    ----------
    wavelength  :   numpy array 
                    model wavelength (in Angstroms)

    rv          :   float
                    radial velocity shift (in km/s)

    Returns
    -------
    wavelength  :   numpy array 
                    shifted model wavelength (in Angstroms)
    """
    return wavelength * ( 1 + rv / 299792.458)

def applyTelluric(model, tell_alpha=1.0, airmass=1.5, pwv=0.5, altitude=1283, instrument='nirspec'):
    """
    Apply the telluric model on the science model.

    Parameters
    ----------
    model   :   model object
                BT Settl model
    alpha   :   float
                telluric scaling factor (the power on the flux)

    Returns
    -------
    model   :   model object
                BT Settl model times the corresponding model

    """
    # read in a telluric model
    wavelow  = model.wave[0] - 10
    wavehigh = model.wave[-1] + 10
    #telluric_model = smart.getTelluric(wavelow=wavelow, wavehigh=wavehigh, alpha=alpha, airmass=airmass)

    telluric_model = smart.Model()
    if instrument.lower() == 'apf':
        telluric_model.wave, telluric_model.flux =  smart.InterpTelluricModel(wavelow=wavelow, wavehigh=wavehigh, altitude=altitude, pwv=pwv, tellset='psg')
    else:
        telluric_model.wave, telluric_model.flux =  smart.InterpTelluricModel(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)

    # apply the telluric alpha parameter
    telluric_model.flux = telluric_model.flux**(tell_alpha)

    #if len(model.wave) > len(telluric_model.wave):
    #   print("The model has a higher resolution ({}) than the telluric model ({})."\
    #       .format(len(model.wave),len(telluric_model.wave)))
    #   model.flux = np.array(smart.integralResample(xh=model.wave, 
    #       yh=model.flux, xl=telluric_model.wave))
    #   model.wave = telluric_model.wave
    #   model.flux *= telluric_model.flux

    #elif len(model.wave) < len(telluric_model.wave):
    ## This should be always true
    telluric_model.flux = np.array(smart.integralResample(xh=telluric_model.wave, yh=telluric_model.flux, xl=model.wave))
    telluric_model.wave = model.wave
    model.flux         *= telluric_model.flux

    #elif len(model.wave) == len(telluric_model.wave):
    #   model.flux *= telluric_model.flux
        
    return model

def convolveTelluric(lsf, telluric_data, alpha=1.0, airmass='1.0', pwv='1.5'):
    """
    Return a convolved telluric transmission model given a telluric data and lsf.
    """
    # get a telluric standard model
    wavelow               = telluric_data.wave[0]  - 50
    wavehigh              = telluric_data.wave[-1] + 50
    telluric_model        = smart.getTelluric(wavelow=wavelow,wavehigh=wavehigh, airmass=airmass, pwv=pwv)
    telluric_model.flux **= alpha
    # lsf
    telluric_model.flux = smart.broaden(wave=telluric_model.wave, flux=telluric_model.flux, 
        vbroad=lsf, rotate=False, gaussian=True)
    # resample
    telluric_model.flux = np.array(smart.integralResample(xh=telluric_model.wave, 
        yh=telluric_model.flux, xl=telluric_data.wave))
    telluric_model.wave = telluric_data.wave
    return telluric_model

def getLSF2(telluric_data, continuum=True, test=False, save_path=None):
    """
    Return a best LSF value from a telluric data.
    """
    
    data = copy.deepcopy(telluric_data)

    def bestParams(data, i, alpha, c2, c0):

        data2          = copy.deepcopy(data)
        data2.wave     = data2.wave + c0
        telluric_model = smart.convolveTelluric(i, data2, alpha=alpha)
        model          = smart.continuum(data=data2, mdl=telluric_model)
        #plt.figure(2)
        #plt.plot(model.wave, model.flux+c2, 'r-', alpha=0.5)
        #plt.plot(data.wave*c1+c0, data.flux, 'b-', alpha=0.5)
        #plt.close()
        #plt.show()
        #sys.exit()
        return model.flux + c2

    def bestParams2(theta, data):

        i, alpha, c2, c0, c1 = theta 
        data2                = copy.deepcopy(data)
        data2.wave           = data2.wave*c1 + c0
        telluric_model       = smart.convolveTelluric(i, data2, alpha=alpha)
        model                = smart.continuum(data=data2, mdl=telluric_model)
        return np.sum(data.flux - (model.flux + c2))**2

    from scipy.optimize import curve_fit, minimize

    popt, pcov = curve_fit(bestParams, data, data.flux, p0=[4.01, 1.01, 0.01, 1.01], maxfev=1000000, epsfcn=0.1)

    #nll = lambda *args: bestParams2(*args)
    #results = minimize(nll, [3., 1., 0.1, -10., 1.], args=(data))
    #popt = results['x']

    data.wave      = data.wave+popt[3]

    telluric_model = smart.convolveTelluric(popt[0], data, alpha=popt[1])
    model          = smart.continuum(data=data, mdl=telluric_model)

    #model.flux * np.e**(-popt[2]) + popt[3]
    model.flux + popt[2]

    return popt[0]

def getLSF(telluric_data, alpha=1.0, continuum=True,test=False,save_path=None):
    """
    Return a best LSF value from a telluric data.
    """
    lsf_list = []
    test_lsf = np.arange(3.0,13.0,0.1)
    
    data = copy.deepcopy(telluric_data)
    if continuum is True:
        data = smart.continuumTelluric(data=data)

    data.flux **= alpha
    for i in test_lsf:
        telluric_model = smart.convolveTelluric(i,data)
        if telluric_data.order == 59:
            telluric_model.flux **= 3
            # mask hydrogen absorption feature
            data2          = copy.deepcopy(data)
            tell_mdl       = copy.deepcopy(telluric_model)
            mask_pixel     = 450
            data2.wave     = data2.wave[mask_pixel:]
            data2.flux     = data2.flux[mask_pixel:]
            data2.noise    = data2.noise[mask_pixel:]
            tell_mdl.wave  = tell_mdl.wave[mask_pixel:]
            tell_mdl.flux  = tell_mdl.flux[mask_pixel:]

            chisquare = smart.chisquare(data2,tell_mdl)

        else:
            chisquare = smart.chisquare(data,telluric_model)
        lsf_list.append([chisquare,i])

        if test is True:
            plt.plot(telluric_model.wave,telluric_model.flux+(i-3)*10+1,
                'r-',alpha=0.5)

    if test is True:
        plt.plot(data.wave,data.flux,
            'k-',label='telluric data',alpha=0.5)
        plt.title("Test LSF",fontsize=15)
        plt.xlabel("Wavelength ($\AA$)",fontsize=12)
        plt.ylabel("Transmission + Offset",fontsize=12)
        plt.minorticks_on()
        if save_path is not None:
            plt.savefig(save_path+\
                "/{}_O{}_lsf_data_mdl.png"\
                .format(data.name, data.order))
        #plt.show()
        plt.close()

        fig, ax = plt.subplots()
        for i in range(len(lsf_list)):
            ax.plot(lsf_list[i][1],lsf_list[i][0],'k.',alpha=0.5)
        ax.plot(min(lsf_list)[1],min(lsf_list)[0],'r.',
            label="best LSF {} km/s".format(min(lsf_list)[1]))
        ax.set_xlabel("LSF (km/s)",fontsize=12)
        ax.set_ylabel("$\chi^2$",fontsize=11)
        plt.minorticks_on()
        plt.legend(fontsize=10)
        if save_path is not None:
            plt.savefig(save_path+\
                "/{}_O{}_lsf_chi2.png"\
                .format(data.name, data.order))
        #plt.show()
        plt.close()

    lsf = min(lsf_list)[1]

    if telluric_data.order == 61 or telluric_data.order == 62 \
    or telluric_data.order == 63: #or telluric_data.order == 64:
        lsf = 5.5
        print("The LSF is obtained from orders 60 and 65 (5.5 km/s).")

    return lsf

def getAlpha(telluric_data,lsf,continuum=True,test=False,save_path=None):
    """
    Return a best alpha value from a telluric data.
    """
    alpha_list = []
    test_alpha = np.arange(0.1,7,0.1)

    data = copy.deepcopy(telluric_data)
    if continuum is True:
        data = smart.continuumTelluric(data=data)

    for i in test_alpha:
        telluric_model = smart.convolveTelluric(lsf,data,
            alpha=i)
        #telluric_model.flux **= i 
        if data.order == 59:
            # mask hydrogen absorption feature
            data2          = copy.deepcopy(data)
            tell_mdl       = copy.deepcopy(telluric_model)
            mask_pixel     = 450
            data2.wave     = data2.wave[mask_pixel:]
            data2.flux     = data2.flux[mask_pixel:]
            data2.noise    = data2.noise[mask_pixel:]
            tell_mdl.wave  = tell_mdl.wave[mask_pixel:]
            tell_mdl.flux  = tell_mdl.flux[mask_pixel:]

            chisquare = smart.chisquare(data2,tell_mdl)

        else:
            chisquare = smart.chisquare(data,telluric_model)
        alpha_list.append([chisquare,i])

        if test is True:
            plt.plot(telluric_model.wave,telluric_model.flux+i*10,
                'k-',alpha=0.5)

    if test is True:
        plt.plot(telluric_data.wave,telluric_data.flux,
            'r-',alpha=0.5)
        plt.rc('font', family='sans-serif')
        plt.title("Test Alpha",fontsize=15)
        plt.xlabel("Wavelength ($\AA$)",fontsize=12)
        plt.ylabel("Transmission + Offset",fontsize=12)
        plt.minorticks_on()
        if save_path is not None:
            plt.savefig(save_path+\
                "/{}_O{}_alpha_data_mdl.png"\
                .format(telluric_data.name,
                    telluric_data.order))
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        plt.rc('font', family='sans-serif')
        for i in range(len(alpha_list)):
            ax.plot(alpha_list[i][1],alpha_list[i][0],'k.',alpha=0.5)
        ax.plot(min(alpha_list)[1],min(alpha_list)[0],'r.',
            label="best alpha {}".format(min(alpha_list)[1]))
        ax.set_xlabel(r"$\alpha$",fontsize=12)
        ax.set_ylabel("$\chi^2$",fontsize=12)
        plt.minorticks_on()
        plt.legend(fontsize=10)
        if save_path is not None:
            plt.savefig(save_path+\
                "/{}_O{}_alpha_chi2.png"\
                .format(telluric_data.name,
                    telluric_data.order))
        plt.show()
        plt.close()

    alpha = min(alpha_list)[1]

    return alpha

def getFringeFrequecy(tell_data, test=False):
    """
    Use the Lomb-Scargle Periodogram to identify 
    the fringe pattern.
    """
    tell_sp  = copy.deepcopy(tell_data)

    ## continuum correction
    tell_sp  = smart.continuumTelluric(data=tell_sp, order=tell_sp.order)

    ## get a telluric model
    lsf      = smart.getLSF(tell_sp)
    alpha    = smart.getAlpha(tell_sp,lsf)
    tell_mdl = smart.convolveTelluric(lsf=lsf,
        telluric_data=tell_sp,alpha=alpha)

    ## fit the fringe pattern in the residual
    pgram_x = np.array(tell_sp.wave,float)[10:-10]
    pgram_y = np.array(tell_sp.flux - tell_mdl.flux,float)[10:-10]
    offset  = np.mean(pgram_y)
    pgram_y -= offset
    mask    = np.where(pgram_y - 1.5 * np.absolute(np.std(pgram_y)) > 0)
    pgram_x = np.delete(pgram_x, mask)
    pgram_y = np.delete(pgram_y, mask)
    pgram_x = np.array(pgram_x, float)
    pgram_y = np.array(pgram_y, float)

    #f = np.lismartace(0.01,10,100000)
    f = np.lismartace(1.0,10,100000)

    ## Lomb Scargle Periodogram
    pgram = signal.lombscargle(pgram_x, pgram_y, f)

    if test:
        fig, ax = plt.subplots(figsize=(16,6))
        ax.plot(f,pgram, 'k-', label='residual',alpha=0.5)
        ax.set_xlabel('frequency')
        plt.legend()
        plt.show()
        plt.close()

    return f[np.argmax(pgram)]

def initModelFit(sci_data, lsf, modelset='btsettl08'):
    """
    Conduct simple chisquare fit to obtain the initial parameters
    for the forward modeling MCMC.

    The function would calculate the chisquare for teff, logg, vini, rv, and alpha.

    Parameters
    ----------
    data                :   spectrum object
                            input science data

    lsf                 :   float
                            line spread function for the NIRSPEC

    Returns
    -------
    best_params_dic     :   dic
                            a dictionary that stores the best parameters for 
                            teff, logg, vsini, rv, and alpha

    chisquare           :   int
                            minimum chisquare

    """
    data            = copy.deepcopy(sci_data)

    ## set up the parameter grid for chisquare computation
    teff_array      = np.arange(1200,3001,100)
    logg_array      = np.arange(3.5,5.51,0.5)
    vsini_array     = np.arange(10,101,10)
    rv_array        = np.arange(-200,201,50)
    alpha_array     = np.arange(0.5,2.01,0.5)
    chisquare_array = np.empty(len(teff_array)*len(logg_array)*len(vsini_array)*len(rv_array)*len(alpha_array))\
    .reshape(len(teff_array),len(logg_array),len(vsini_array),len(rv_array),len(alpha_array))

    time1 = time.time()
    for i, teff in enumerate(teff_array):
        for j, logg in enumerate(logg_array):
            for k, vsini in enumerate(vsini_array):
                for l, rv in enumerate(rv_array):
                    for m, alpha in enumerate(alpha_array):
                        model = smart.makeModel(teff, logg, 0.0, vsini, rv, alpha, 0, 0,
                            lsf=lsf, order=str(data.order), data=data, modelset=modelset)
                        chisquare_array[i,j,k,l,m] = smart.chisquare(data, model)
    time2 = time.time()
    print("total time:",time2-time1)

    ind = np.unravel_index(np.argmin(chisquare_array, axis=None), chisquare_array.shape)
    print("ind ",ind)
    chisquare       = chisquare_array[ind]

    best_params_dic = {'teff':teff_array[ind[0]], 'logg':logg_array[ind[1]], 
    'vsini':vsini_array[ind[2]], 'rv':rv_array[ind[3]], 'alpha':alpha_array[ind[4]]}

    print(best_params_dic, chisquare)

    return best_params_dic , chisquare

