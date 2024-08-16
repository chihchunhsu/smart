#!/usr/bin/env python
import numpy as np
from astropy.io import fits
from astropy.table import Table
import smart
#import fringe_model
import sys, os, os.path, time
import copy

def GetModel(wavelow, wavehigh, method='pwv', wave=False, **kwargs):
    """
    Get a telluric spectrum using the atmosphere models in Moehler et al. (2014).

    Parameters
    ----------
    wavelow     :   int
                    lower bound of the wavelength range

    wavehigh    :   int
                    upper bound of the wavelength range

    Optional Parameters
    -------------------
    airmass     :   str
                    airmass of the telluric model, either 1.0 or 1.5
    
    alpha       :   float
                    the power alpha parameter of the telluric model

    method      :   str
                    'pwv' or 'season'
                    
                    The defulat method is 'pwv', with airmasses 1.0, 1.5, 2.0, 2.5, 3.0, 
                    and PWV (in mm) of 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, and 20.0

                    Another method is 'season', with airmasses 1.0, 1.5, 2.0, 2.5, 3.0, 
                    and bi-monthly average PWV values (1 = December/January ...6 = October/November)


    Returns
    -------
    telluric: model object
              a telluric model with wavelength and flux


    Examples
    --------
    >>> import smart
    >>> telluric = smart.getTelluric(wavelow=22900, wavehigh=23250)

    """
    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    airmass = kwargs.get('airmass', 1.5)
    alpha   = kwargs.get('alpha', 1.0)
    # keyword argument for pwv
    pwv     = kwargs.get('pwv', 0.5)
    # keyword argument for season
    season  = kwargs.get('season', 0)

    airmass_str = str(int(10*airmass))
    pwv_str = str(int(10*pwv)).zfill(3)

    if method == 'pwv':
        tfile = BASE + '/../libraries/telluric/pwv_R300k_airmass{}/LBL_A{}_s0_w{}_R0300000_T.fits'.format(airmass, 
            airmass_str, pwv_str)

    #elif method == 'season':
    #   tfile = '/../libraries/telluric/season_R300k_airmass{}/LBL_A{}_s{}_R0300000_T.fits'.format(airmass, 
    #       airmass_str, season_str)
    
    tellurics = fits.open(tfile)

    telluric      = smart.Model()
    telluric.wave = np.array(tellurics[1].data['lam'] * 10000) # convert to Angstrom
    telluric.flux = np.array(tellurics[1].data['trans'])**(alpha)

    # select the wavelength range
    criteria      = (telluric.wave > wavelow) & (telluric.wave < wavehigh)

    telluric.wave = telluric.wave[criteria]
    telluric.flux = telluric.flux[criteria]

    if wave:
        return telluric.wave
    else:
        return telluric.flux

def InterpTelluricModel(wavelow, wavehigh, airmass, pwv):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    Gridfile = BASE + '/../libraries/telluric/pwv_R300k_gridparams.csv'

    T1 = Table.read(Gridfile)

    # Check if the model already exists (grid point)
    if (airmass, pwv) in zip(T1['airmass'], T1['pwv']):
        flux2  = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == airmass) & (T1['pwv'] == pwv))][0], pwv=T1['pwv'][np.where((T1['airmass'] == airmass) & (T1['pwv'] == pwv))][0])
        waves2 = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == airmass) & (T1['pwv'] == pwv))][0], pwv=T1['pwv'][np.where((T1['airmass'] == airmass) & (T1['pwv'] == pwv))][0], wave=True)
        return waves2, flux2

    # Get the nearest models to the gridpoint (airmass)
    x1 = np.max(T1['airmass'][np.where(T1['airmass'] <= airmass)])
    x2 = np.min(T1['airmass'][np.where(T1['airmass'] >= airmass)])
    y1 = np.max(list(set(T1['pwv'][np.where( ( (T1['airmass'] == x1) & (T1['pwv'] <= pwv) ) )]) & set(T1['pwv'][np.where( ( (T1['airmass'] == x2) & (T1['pwv'] <= pwv) ) )])))
    y2 = np.min(list(set(T1['pwv'][np.where( ( (T1['airmass'] == x1) & (T1['pwv'] >= pwv) ) )]) & set(T1['pwv'][np.where( ( (T1['airmass'] == x2) & (T1['pwv'] >= pwv) ) )])))

    # Check if the gridpoint exists within the model ranges
    for x in [x1, x2]:
        for y in [y1, y2]:
            if (x, y) not in zip(T1['airmass'], T1['pwv']):
                print('No Model', x, y)
                return 1
    
    # Get the four points
    Points =  [ 
                [T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y1))].data[0], T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y1))].data[0], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y1))][0]))],
                [T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y2))].data[0], T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y2))].data[0], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y2))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y2))][0]))],
                [T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y1))].data[0], T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y1))].data[0], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y1))][0]))],
                [T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y2))].data[0], T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y2))].data[0], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y2))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y2))][0]))],
              ]

    waves2 = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y1))][0], wave=True)

    return waves2, smart.utils.interpolations.bilinear_interpolation(airmass, pwv, Points)

def convolveTelluric(lsf, airmass, pwv, telluric_data):
    """
    Return a convolved, normalized telluric transmission model given a telluric data and lsf.
    """
    # get a telluric standard model
    wavelow               = telluric_data.wave[0]  - 50
    wavehigh              = telluric_data.wave[-1] + 50
    modelwave, modelflux  = InterpTelluricModel(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
    #modelflux           **= alpha
    # lsf
    modelflux             = smart.broaden(wave=modelwave, flux=modelflux, vbroad=lsf, rotate=False, gaussian=True)
    # resample
    modelflux             = np.array(smart.integralResample(xh=modelwave, yh=modelflux, xl=telluric_data.wave))
    modelwave             = telluric_data.wave
    telluric_model        = smart.Model()
    telluric_model.flux   = modelflux
    telluric_model.wave   = modelwave

    return telluric_model

def makeTelluricModel(lsf, airmass, pwv, flux_offset, wave_offset, data, deg=2, niter=None, **kwargs):
    """
    Make a continuum-corrected telluric model as a function of LSF, airmass, pwv, and flux and wavelength offsets.

    The model assumes a second-order polynomail for the continuum.
    """
    include_fringe_model = kwargs.get('include_fringe_model', False)

    if include_fringe_model:
        piece_wise_fringe_model_list = kwargs.get('piece_wise_fringe_model_list', [0, 700, -700, -1])
        s1, s2, s3, s4 = piece_wise_fringe_model_list

        # for single order, (three) piece-wise Fabry Perot fringe model
        A1       = kwargs.get('A1') 
        A2       = kwargs.get('A2') 
        A3       = kwargs.get('A3') 
        Dos1     = kwargs.get('Dos1') 
        Dos2     = kwargs.get('Dos2') 
        Dos3     = kwargs.get('Dos3') 
        R1       = kwargs.get('R1') 
        R2       = kwargs.get('R2') 
        R3       = kwargs.get('R3') 
        phi1     = kwargs.get('phi1') 
        phi2     = kwargs.get('phi2') 
        phi3     = kwargs.get('phi3') 

    data2               = copy.deepcopy(data)
    data2.wave          = data2.wave + wave_offset
    telluric_model      = convolveTelluric(lsf, airmass, pwv, data2)
    
    model               = smart.continuum(data=data2, mdl=telluric_model, deg=deg)
    if niter is not None:
        for i in range(niter):
            model               = smart.continuum(data=data2, mdl=model, deg=deg)

    model.flux         += flux_offset

    if include_fringe_model:
        popt1 = np.array([A1, Dos1, R1, phi1])
        popt2 = np.array([A2, Dos2, R2, phi2])
        popt3 = np.array([A3, Dos3, R3, phi3])

        model.flux[s1:s2] = model.flux[s1:s2]*(1+smart.forward_model.fringe_model.Fabry_Perot_zero(data2.wave[s1:s2], *popt1))
        model.flux[s2:s3] = model.flux[s2:s3]*(1+smart.forward_model.fringe_model.Fabry_Perot_zero(data2.wave[s2:s3], *popt2))
        model.flux[s3:s4] = model.flux[s3:s4]*(1+smart.forward_model.fringe_model.Fabry_Perot_zero(data2.wave[s3:s4], *popt3))

    return model

def convolveTelluricFringe(lsf, airmass, pwv, telluric_data, 
    a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, 
    a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, 
    a1_3, k1_3, p1_3, a2_3, k2_3, p2_3):
    """
    Return a convolved, normalized telluric transmission model given a telluric data and lsf.

    The telluric model is dependent on airmass and precipitable water vapor (pwv), and a fringe model
    as a function of wavelength dependent fringe amplitude, wave number, and phase (mostly constant).

    The wavelength dependent fringe model is approximately three chunks, currently set as the first 29 percent 
    and the last 21 percent of the pixels (s1, s2, s3, and s4).

    """
    # get a telluric standard model
    wavelow               = telluric_data.wave[0]  - 50
    wavehigh              = telluric_data.wave[-1] + 50
    modelwave, modelflux  = InterpTelluricModel(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
    #modelflux           **= alpha

    # define three piece-wise slices to model the fringe pattern (wavelength-dependent wave number k)
    s1, s2, s3, s4 = 0, int(len(modelflux)*0.29), -int(len(modelflux)*0.21), -1

    # apply fringe correction
    modelflux[s1:s2] = modelflux[s1:s2]*(1+smart.double_sine(wave=modelwave[s1:s2], a1=a1_1, k1=k1_1, p1=p1_1, a2=a2_1, k2=k2_1, p2=p2_1))
    modelflux[s2:s3] = modelflux[s2:s3]*(1+smart.double_sine(wave=modelwave[s2:s3], a1=a1_2, k1=k1_2, p1=p1_2, a2=a2_2, k2=k2_2, p2=p2_2))
    modelflux[s3:s4] = modelflux[s3:s4]*(1+smart.double_sine(wave=modelwave[s3:s4], a1=a1_3, k1=k1_3, p1=p1_3, a2=a2_3, k2=k2_3, p2=p2_3))

    # lsf
    modelflux             = smart.broaden(wave=modelwave, flux=modelflux, vbroad=lsf, rotate=False, gaussian=True)
    # resample
    modelflux             = np.array(smart.integralResample(xh=modelwave, yh=modelflux, xl=telluric_data.wave))
    modelwave             = telluric_data.wave
    telluric_model        = smart.Model()
    telluric_model.flux   = modelflux
    telluric_model.wave   = modelwave

    return telluric_model

def convolveTelluricFringe(lsf, airmass, pwv, telluric_data, 
    a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, 
    a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, 
    a1_3, k1_3, p1_3, a2_3, k2_3, p2_3):
    """
    Return a convolved, normalized telluric transmission model given a telluric data and lsf.

    The telluric model is dependent on airmass and precipitable water vapor (pwv), and a fringe model
    as a function of wavelength dependent fringe amplitude, wave number, and phase (mostly constant).

    The wavelength dependent fringe model is approximately three chunks, currently set as the first 29 percent 
    and the last 21 percent of the pixels (s1, s2, s3, and s4).

    """
    # get a telluric standard model
    wavelow               = telluric_data.wave[0]  - 50
    wavehigh              = telluric_data.wave[-1] + 50
    modelwave, modelflux  = InterpTelluricModel(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
    #modelflux           **= alpha

    # define three piece-wise slices to model the fringe pattern (wavelength-dependent wave number k)
    s1, s2, s3, s4 = 0, int(len(modelflux)*0.29), -int(len(modelflux)*0.21), -1

    # apply fringe correction
    modelflux[s1:s2] = modelflux[s1:s2]*(1+smart.double_sine(wave=modelwave[s1:s2], a1=a1_1, k1=k1_1, p1=p1_1, a2=a2_1, k2=k2_1, p2=p2_1))
    modelflux[s2:s3] = modelflux[s2:s3]*(1+smart.double_sine(wave=modelwave[s2:s3], a1=a1_2, k1=k1_2, p1=p1_2, a2=a2_2, k2=k2_2, p2=p2_2))
    modelflux[s3:s4] = modelflux[s3:s4]*(1+smart.double_sine(wave=modelwave[s3:s4], a1=a1_3, k1=k1_3, p1=p1_3, a2=a2_3, k2=k2_3, p2=p2_3))

    # lsf
    modelflux             = smart.broaden(wave=modelwave, flux=modelflux, vbroad=lsf, rotate=False, gaussian=True)
    # resample
    modelflux             = np.array(smart.integralResample(xh=modelwave, yh=modelflux, xl=telluric_data.wave))
    modelwave             = telluric_data.wave
    telluric_model        = smart.Model()
    telluric_model.flux   = modelflux
    telluric_model.wave   = modelwave

    return telluric_model


def makeTelluricModelFringe(lsf, airmass, pwv, flux_offset, wave_offset, 
    a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, 
    a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, 
    a1_3, k1_3, p1_3, a2_3, k2_3, p2_3,
    data, deg=2, niter=None):
    """
    Make a continuum-corrected telluric model as a function of LSF, airmass, pwv, and flux and wavelength offsets.

    The model assumes a second-order polynomail for the continuum.
    """
    data2               = copy.deepcopy(data)
    data2.wave          = data2.wave + wave_offset
    #telluric_model      = convolveTelluric(lsf, airmass, pwv, data2)
    # three piece-wise slices to model the fringe pattern (wavelength-dependent wave number k)
    telluric_model      = convolveTelluricFringe(   lsf, airmass, pwv, data2, 
                                                    a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, 
                                                    a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, 
                                                    a1_3, k1_3, p1_3, a2_3, k2_3, p2_3)

    model               = smart.continuum(data=data2, mdl=telluric_model, deg=deg)

    if niter is not None:
        for i in range(niter):
            model               = smart.continuum(data=data2, mdl=model, deg=deg)
    
    model.flux         += flux_offset

    return model

#### Wavelength dependent amplitude and wavenumber implementation

def convolveTelluricFringeWaveDependent(lsf, airmass, pwv, telluric_data, 
    a1_1, a1_0, k1_1, k1_0, p1, a2_1, a2_0, k2_2, k2_1, k2_0, p2):
    """
    Return a convolved, normalized telluric transmission model given a telluric data and lsf.

    The telluric model is dependent on airmass and precipitable water vapor (pwv), and a fringe model
    as a function of wavelength dependent fringe amplitude, wave number, and phase (mostly constant).

    Amplitudes "a1" and "a2" are modeled as a linear function of wavelength. 
    The wavenumbers "k1" (~2.1 Angstrom) is best described as a lienar function and 
    "k2" (~0.85 Angstrom) is best modeled as a second order polynomial.

    """
    # get a telluric standard model
    wavelow               = telluric_data.wave[0]  - 50
    wavehigh              = telluric_data.wave[-1] + 50
    modelwave, modelflux  = InterpTelluricModel(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
    #modelflux           **= alpha

    # apply fringe correction
    modelflux             = modelflux * (1+smart.doub_sine_wave_dependent(wave=modelwave, a1_1=a1_1, a1_0=a1_0, k1_1=k1_1, k1_0=k1_0, p1=p1, 
                                                                            a2_1=a2_1, a2_0=a2_0, k2_2=k2_2, k2_1=k2_1, k2_0=k2_0, p2=p2))

    # lsf
    modelflux             = smart.broaden(wave=modelwave, flux=modelflux, vbroad=lsf, rotate=False, gaussian=True)
    # resample
    modelflux             = np.array(smart.integralResample(xh=modelwave, yh=modelflux, xl=telluric_data.wave))
    modelwave             = telluric_data.wave
    telluric_model        = smart.Model()
    telluric_model.flux   = modelflux
    telluric_model.wave   = modelwave

    return telluric_model

def makeTelluricModelFringeWaveDependent(lsf, airmass, pwv, flux_offset, wave_offset, 
    a1_1, a1_0, k1_1, k1_0, p1, a2_1, a2_0, k2_2, k2_1, k2_0, p2,
    data, deg=2, niter=None):
    """
    Make a continuum-corrected telluric model as a function of LSF, airmass, pwv, and flux and wavelength offsets.

    The model assumes a second-order polynomail for the continuum.
    """
    data2               = copy.deepcopy(data)
    data2.wave          = data2.wave + wave_offset
    #telluric_model      = convolveTelluric(lsf, airmass, pwv, data2)
    # apply fringe correction
    telluric_model      = convolveTelluricFringeWaveDependent(   lsf, airmass, pwv, data2, 
                                                    a1_1, a1_0, k1_1, k1_0, p1, a2_1, a2_0, k2_2, k2_1, k2_0, p2)

    model               = smart.continuum(data=data2, mdl=telluric_model, deg=deg)

    if niter is not None:
        for i in range(niter):
            model               = smart.continuum(data=data2, mdl=model, deg=deg)
    
    model.flux         += flux_offset

    return model

