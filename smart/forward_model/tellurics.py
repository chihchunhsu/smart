#!/usr/bin/env python
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import smart
import sys, os, os.path, time
import copy

def GetModel(wavelow, wavehigh, method='pwv', wave=False, tellset='moehler2014', **kwargs):
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

    airmass    = kwargs.get('airmass', 1.5)
    alpha      = kwargs.get('alpha', 1.0)
    # keyword argument for pwv
    pwv        = kwargs.get('pwv', 0.5)
    # keyword argument for season
    season     = kwargs.get('season', 0)
    # keyword argument for altitude
    altitude   = kwargs.get('altitude', 0)
    instrument = kwargs.get('instrument', 'nirspec')
    order      = kwargs.get('order', 'O33')

    if tellset == 'moehler2014':

        airmass_str = str(int(10*airmass))
        pwv_str     = str(int(10*pwv)).zfill(3)

        if method == 'pwv':
            tfile = BASE + '/../libraries/telluric/pwv_R300k_airmass{}/LBL_A{}_s0_w{}_R0300000_T.fits'.format(airmass, 
                airmass_str, pwv_str)

        #elif method == 'season':
        #   tfile = '/../libraries/telluric/season_R300k_airmass{}/LBL_A{}_s{}_R0300000_T.fits'.format(airmass, 
        #       airmass_str, season_str)
        
        tellurics     = fits.open(tfile)

        telluric      = smart.Model()
        telluric.wave = np.array(tellurics[1].data['lam'] * 10000) # convert to Angstrom
        telluric.flux = np.array(tellurics[1].data['trans'])**(alpha)


    if tellset == 'psg':
        
        alt_key  = {0  :'00', 2600:'01', 4200:'02', 14000:'03', 35000:'04'}
        pwv_key  = {0.5:'00', 1.5 :'01', 3.5 :'02', 5.0  :'03'}

        altitude_str = alt_key[altitude]
        pwv_str      = pwv_key[pwv]

        #tfile = BASE + '/../libraries/telluric/psg_telluric/tel_3500_10500_{}_{}.txt'.format(altitude_str, pwv_str)

        if instrument.lower() == 'apf':
            tfile = BASE + '/../libraries/telluric/psg_telluric/{}/{}/tel_3500_10500_{}_{}.fits'.format(instrument.upper(), order.upper(), altitude_str, pwv_str)

        #tellurics     = ascii.read(tfile, format='fast_basic', comment='#', delimiter=' ', names=['col1','col2'])
        #tellurics     = Table.read(tfile)
        tellurics      = fits.open(tfile)
 
        telluric       = smart.Model()
        #telluric.wave  = np.array(tellurics['col1']) # in Angstroms already
        #telluric.flux  = np.array(tellurics['col2'])**(alpha)
        telluric.wave  = np.array(tellurics[1].data.field(0)) # in Angstroms already
        telluric.flux  = np.array(tellurics[1].data.field(1))**(alpha)

    # select the wavelength range
    criteria      = (telluric.wave > wavelow) & (telluric.wave < wavehigh)

    telluric.wave = telluric.wave[criteria]
    telluric.flux = telluric.flux[criteria]

    if wave:
        return telluric.wave
    else:
        return telluric.flux

def InterpTelluricModel(wavelow, wavehigh, tellset='moehler2014', **kwargs):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    airmass    = kwargs.get('airmass', 1.5)
    pwv        = kwargs.get('pwv', 1.5)
    altitude   = kwargs.get('altitude', 1283)
    instrument = kwargs.get('instrument', 'nirspec')
    order      = kwargs.get('order', 'O33')

    if tellset == 'moehler2014':

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
                    [T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y1))], T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y1))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y1))][0]))],
                    [T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y2))], T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y2))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y2))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y2))][0]))],
                    [T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y1))], T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y1))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y1))][0]))],
                    [T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y2))], T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y2))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x2) & (T1['pwv'] == y2))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x2) & (T1['pwv'] == y2))][0]))],
                  ]

        waves2 = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['airmass'][np.where( (T1['airmass'] == x1) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['airmass'] == x1) & (T1['pwv'] == y1))][0], wave=True)

    if tellset == 'psg':

        Gridfile = BASE + '/../libraries/telluric/psg_telluric/psg_telluric_gridparams.csv'

        T1 = Table.read(Gridfile)

        altitude = 1283 # need to make this a keyword for different observatories

        # Check if the model already exists (grid point)
        if (airmass, pwv) in zip(T1['altitude'], T1['pwv']):
            flux2  = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['altitude'][np.where( (T1['altitude'] == altitude) & (T1['pwv'] == pwv))][0], pwv=T1['pwv'][np.where((T1['altitude'] == altitude) & (T1['pwv'] == pwv))][0], tellset='psg', instrument=instrument, order=order)
            waves2 = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['altitude'][np.where( (T1['altitude'] == altitude) & (T1['pwv'] == pwv))][0], pwv=T1['pwv'][np.where((T1['altitude'] == altitude) & (T1['pwv'] == pwv))][0], wave=True, tellset='psg', instrument=instrument, order=order)
            return waves2, flux2

        # Get the nearest models to the gridpoint (airmass)
        x1 = np.max(T1['altitude'][np.where(T1['altitude'] <= altitude)])
        x2 = np.min(T1['altitude'][np.where(T1['altitude'] >= altitude)])
        y1 = np.max(list(set(T1['pwv'][np.where( ( (T1['altitude'] == x1) & (T1['pwv'] <= pwv) ) )]) & set(T1['pwv'][np.where( ( (T1['altitude'] == x2) & (T1['pwv'] <= pwv) ) )])))
        y2 = np.min(list(set(T1['pwv'][np.where( ( (T1['altitude'] == x1) & (T1['pwv'] >= pwv) ) )]) & set(T1['pwv'][np.where( ( (T1['altitude'] == x2) & (T1['pwv'] >= pwv) ) )])))

        # Check if the gridpoint exists within the model ranges
        for x in [x1, x2]:
            for y in [y1, y2]:
                if (x, y) not in zip(T1['altitude'], T1['pwv']):
                    print('No Model', x, y)
                    return 1
        
        # Get the four points
        Points =  [ 
                    [T1['altitude'][np.where( (T1['altitude'] == x1) & (T1['pwv'] == y1))], T1['pwv'][np.where((T1['altitude'] == x1) & (T1['pwv'] == y1))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, altitude=T1['altitude'][np.where( (T1['altitude'] == x1) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['altitude'] == x1) & (T1['pwv'] == y1))][0], tellset=tellset, instrument=instrument, order=order))],
                    [T1['altitude'][np.where( (T1['altitude'] == x1) & (T1['pwv'] == y2))], T1['pwv'][np.where((T1['altitude'] == x1) & (T1['pwv'] == y2))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, altitude=T1['altitude'][np.where( (T1['altitude'] == x1) & (T1['pwv'] == y2))][0], pwv=T1['pwv'][np.where((T1['altitude'] == x1) & (T1['pwv'] == y2))][0], tellset=tellset, instrument=instrument, order=order))],
                    [T1['altitude'][np.where( (T1['altitude'] == x2) & (T1['pwv'] == y1))], T1['pwv'][np.where((T1['altitude'] == x2) & (T1['pwv'] == y1))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, altitude=T1['altitude'][np.where( (T1['altitude'] == x2) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['altitude'] == x2) & (T1['pwv'] == y1))][0], tellset=tellset, instrument=instrument, order=order))],
                    [T1['altitude'][np.where( (T1['altitude'] == x2) & (T1['pwv'] == y2))], T1['pwv'][np.where((T1['altitude'] == x2) & (T1['pwv'] == y2))], np.log10(GetModel(wavelow=wavelow, wavehigh=wavehigh, altitude=T1['altitude'][np.where( (T1['altitude'] == x2) & (T1['pwv'] == y2))][0], pwv=T1['pwv'][np.where((T1['altitude'] == x2) & (T1['pwv'] == y2))][0], tellset=tellset, instrument=instrument, order=order))],
                  ]

        waves2 = GetModel(wavelow=wavelow, wavehigh=wavehigh, airmass=T1['altitude'][np.where( (T1['altitude'] == x1) & (T1['pwv'] == y1))][0], pwv=T1['pwv'][np.where((T1['altitude'] == x1) & (T1['pwv'] == y1))][0], wave=True, tellset=tellset, instrument=instrument, order=order)

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

def makeTelluricModel(lsf, airmass, pwv, flux_offset, wave_offset, data, deg=2, niter=None):
    """
    Make a continuum-corrected telluric model as a function of LSF, airmass, pwv, and flux and wavelength offsets.

    The model assumes a second-order polynomail for the continuum.
    """
    data2               = copy.deepcopy(data)
    data2.wave          = data2.wave + wave_offset
    telluric_model      = convolveTelluric(lsf, airmass, pwv, data2)
    
    model               = smart.continuum(data=data2, mdl=telluric_model, deg=deg)
    if niter is not None:
        for i in range(niter):
            model               = smart.continuum(data=data2, mdl=model, deg=deg)
    
    model.flux         += flux_offset

    return model

