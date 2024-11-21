import smart
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from astropy.io import fits
from numpy.linalg import inv, det
from ..utils.interpolations import trilinear_interpolation

##############################################################################################################


def InterpModel(teff, logg=4, metal=0, alpha=0, kzz=0, co=0, modelset='phoenix-aces-agss-cond-2011', instrument='nirspec', order='O33'):
    #print('Parameters', teff, logg, modelset, instrument, order)
    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set and instrument
    if instrument.lower() in ['nirspec', 'hires', 'igrins']:
        path     = BASE + '/../libraries/%s/%s-O%s/'%(smart.ModelSets[modelset.lower()], instrument.upper(), str(order).upper().strip('Oo'))
    else:
        path     = BASE + '/../libraries/%s/%s-%s/'%(smart.ModelSets[modelset.lower()], instrument.upper(), order.upper())
    Gridfile = BASE + '/../libraries/%s/%s_gridparams.csv'%(smart.ModelSets[modelset.lower()], smart.ModelSets[modelset.lower()])

    if modelset.lower() == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'

    # Read the grid file
    T1 = Table.read(Gridfile)
    #print(T1)

    ###################################################################################

    def GetModel(temp, wave=False, **kwargs):
        
        logg       = kwargs.get('logg', 4.5)
        metal      = kwargs.get('metal', 0)
        alpha      = kwargs.get('alpha', 0)
        kzz        = kwargs.get('kzz', 0)
        co         = kwargs.get('co', 0)
        gridfile   = kwargs.get('gridfile', None)
        instrument = kwargs.get('instrument', 'nirspec')
        order      = kwargs.get('order', None)
        #print(temp, logg, metal, alpha)
        if gridfile is None:
            raise ValueError('Model gridfile must be provided.') 
        
        if modelset.lower() == 'btsettl08' and instrument.lower() == 'nirspec': 
            filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'    
        elif 'sonora' in modelset.lower() and '2023' not in modelset:
            if instrument.lower() == 'nirspec':
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_FeH0.00_Y0.28_CO1.00' + '_%s-O%s.fits'%(instrument.upper(), order.upper())
            else:
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_FeH0.00_Y0.28_CO1.00' + '_%s-%s.fits'%(instrument.upper(), order.upper())
        elif modelset.lower() == 'sonora-2023':
            if instrument.lower() == 'nirspec':
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_CO{0:.2f}'.format(float(co)) + '_kzz{0:.2f}'.format(float(kzz)) + '_%s-O%s.fits'%(instrument.upper(), order.upper())
            else:
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_CO{0:.2f}'.format(float(co)) + '_kzz{0:.2f}'.format(float(kzz)) + '_%s-%s.fits'%(instrument.upper(), order.upper())
        elif modelset.lower() == 'marcs-apogee-dr15':
            cm      = kwargs.get('cm', 0)
            nm      = kwargs.get('nm', 0) 
            filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_cm{0:.2f}'.format(float(cm)) + '_nm{0:.2f}'.format(float(nm)) + '_%s-%s.fits'%(instrument.upper(), order.upper())
        elif kzz != 0:
            filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_kzz{0:.2f}'.format(float(kzz)) + '_%s-%s.fits'%(instrument.upper(), order.upper())
        else: 
            if instrument.lower()in ['nirspec', 'hires', 'igrins']:
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_%s-O%s.fits'%(instrument.upper(), order.upper())
            else:
                if modelset == 'hd206893-qkbbhires':
                    kzz = 1e8
                    filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_kzz{0:.2f}'.format(float(kzz)) + '_%s-%s.fits'%(instrument.upper(), order.upper())
                else:
                    filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_%s-%s.fits'%(instrument.upper(), order.upper())        
        # Read in the model FITS file
        if modelset.lower() == 'btsettl08': 
            Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])
        else: 
            Tab = Table.read(path+filename)

        # Return the model (wave of flux)
        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    ###################################################################################

    # Check if the model already exists (grid point)
    if 'sonora' in modelset.lower() and '2023' not in modelset:
        if (teff, logg) in zip(T1['teff'], T1['logg']):
            metal, ys = 0, 0.28
            index0 = np.where( (T1['teff'] == teff) & (T1['logg'] == logg) & (T1['FeH'] == metal) & (T1['Y'] == ys) )
            #flux2  = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset )
            #waves2 = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
            return waves2, flux2

    elif modelset.lower() == 'sonora-2023':
        if (teff, logg, metal, kzz, co) in zip(T1['teff'], T1['logg'], T1['M_H'], T1['kzz'], T1['CO']):
            index0 = np.where( (T1['teff'] == teff) & (T1['logg'] == logg) & (T1['M_H'] == metal) & (T1['kzz'] == kzz) & (T1['CO'] == co) )
            #flux2  = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset )
            #waves2 = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['M_H'][index0], kzz=T1['kzz'][index0], co=T1['CO'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['M_H'][index0], kzz=T1['kzz'][index0], co=T1['CO'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
            return waves2, flux2

    elif kzz != 0:
        if (teff, logg, metal, alpha, kzz) in zip(T1['teff'], T1['logg'], T1['M_H'], T1['en'], T1['kzz']): 
            index0 = np.where( (T1['teff'] == teff) & (T1['logg'] == logg) & (T1['M_H'] == metal) & (T1['en'] == alpha) & (T1['kzz'] == kzz) )
            #flux2  = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset )
            #waves2 = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['M_H'][index0], alpha=T1['en'][index0], kzz=T1['kzz'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['M_H'][index0], alpha=T1['en'][index0], kzz=T1['kzz'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
            return waves2, flux2
    
    else:
        if (teff, logg, metal, alpha) in zip(T1['teff'], T1['logg'], T1['M_H'], T1['en']): 
            index0 = np.where( (T1['teff'] == teff) & (T1['logg'] == logg) & (T1['M_H'] == metal) & (T1['en'] == alpha) )
            #flux2  = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset )
            #waves2 = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['M_H'][index0], alpha=T1['en'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['M_H'][index0], alpha=T1['en'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
            return waves2, flux2


    try:

        if 'sonora' in modelset.lower() and '2023' not in modelset:

            metal, alpha = 0, 0.28
            # Get the nearest models to the gridpoint (teff)
            x0 = np.max(T1['teff'][np.where(T1['teff'] <= teff)])
            x1 = np.min(T1['teff'][np.where(T1['teff'] >= teff)])
            #print(x0, x1)
            
            # Get the nearest grid point to logg
            y0 = np.max(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] <= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] <= logg) )])))
            y1 = np.min(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] >= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] >= logg) )])))
            #print(y0, y1)
            
            # Get the nearest grid point to [M/H]
            z0 = np.max(list(set(T1['FeH'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] <= metal) )]) & 
                             set(T1['FeH'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] <= metal) )])))
            z1 = np.min(list(set(T1['FeH'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] >= metal) )]) & 
                             set(T1['FeH'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] >= metal) )])))
            #print(z0, z1)
            
            # Get the nearest grid point to Alpha
            t0 = np.max(list(set(T1['Y'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] <= alpha) )]) & 
                             set(T1['Y'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] <= alpha) )])))
            t1 = np.min(list(set(T1['Y'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] >= alpha) )]) & 
                             set(T1['Y'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] >= alpha) )])))
            #print(t0, t1)
        
        elif modelset.lower() == 'sonora-2023':

            #metal, alpha = 0, 0.28
            # Get the nearest models to the gridpoint (teff)
            x0 = np.max(T1['teff'][np.where(T1['teff'] <= teff)])
            x1 = np.min(T1['teff'][np.where(T1['teff'] >= teff)])
            #print('teff:', x0, teff, x1)
            
            # Get the nearest grid point to logg
            y0 = np.max(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] <= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] <= logg) )])))
            y1 = np.min(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] >= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] >= logg) )])))
            #print('logg:', y0, logg, y1)
            
            # Get the nearest grid point to [M/H]
            z0 = np.max(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] <= metal) )]) & 
                             set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] <= metal) )])))
            z1 = np.min(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] >= metal) )]) & 
                             set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] >= metal) )])))
            #print('metal:', z0, metal, z1)
            
            # Get the nearest grid point to kzz
            t0 = np.max(list(set(T1['kzz'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] <= kzz) )]) & 
                             set(T1['kzz'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] <= kzz) )])))
            t1 = np.min(list(set(T1['kzz'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] >= kzz) )]) & 
                             set(T1['kzz'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] >= kzz) )])))
            #print('kzz:', t0, kzz, t1)

            # Get the nearest grid point to co
            w0 = np.max(list(set(T1['CO'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] <= co) )]) & 
                             set(T1['CO'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] <= co) )])))
            w1 = np.min(list(set(T1['CO'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] >= co) )]) & 
                             set(T1['CO'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] >= co) )])))
            #print('co:', w0, co, w1)

        elif kzz != 0:

            #print('KZZ Models')
            # Get the nearest models to the gridpoint (teff)
            #print(teff, logg, metal, alpha, kzz)
            x0 = np.max(T1['teff'][np.where( (T1['teff'] <= teff) )])
            x1 = np.min(T1['teff'][np.where( (T1['teff'] >= teff) )])
            #print(x0, x1)
            
            # Get the nearest grid point to logg
            y0 = np.max(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] <= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] <= logg) )])))
            y1 = np.min(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] >= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] >= logg) )])))
            #print(y0, y1)
            
            # Get the nearest grid point to [M/H]
            z0 = np.max(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] <= metal) )]) & 
                             set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] <= metal) )])))
            z1 = np.min(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] >= metal) )]) & 
                             set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] >= metal) )])))
            #print(z0, z1)
            
            # Get the nearest grid point to Kzz
            t0 = np.max(list(set(T1['kzz'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] <= kzz) )]) & 
                             set(T1['kzz'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] <= kzz) )])))
            t1 = np.min(list(set(T1['kzz'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] >= kzz) )]) & 
                             set(T1['kzz'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] >= kzz) )])))
            #print(t0, t1)

        else:

            # Get the nearest models to the gridpoint (teff)
            x0 = np.max(T1['teff'][np.where(T1['teff'] <= teff)])
            x1 = np.min(T1['teff'][np.where(T1['teff'] >= teff)])
            #print('teff:', x0, teff, x1)
            # Get the nearest grid point to logg
            y0 = np.max(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] <= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] <= logg) )])))
            y1 = np.min(list(set(T1['logg'][np.where( (T1['teff'] == x0) & (T1['logg'] >= logg) )]) & 
                             set(T1['logg'][np.where( (T1['teff'] == x1) & (T1['logg'] >= logg) )])))
            #print('logg:', y0, logg, y1)
            # Get the nearest grid point to [M/H]
            #print(metal)
            #print(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) )])))
            #print(list(set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) )])))
            #print(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] <= metal))])))
            #print(list(set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] <= metal))])))
            #print(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] >= metal))])))
            #print(list(set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] >= metal))])))
            z0 = np.max(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] <= metal) )]) & 
                             set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] <= metal) )])))
            z1 = np.min(list(set(T1['M_H'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] >= metal) )]) & 
                             set(T1['M_H'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] >= metal) )])))
            #print('metal:', z0, metal, z1)
            # Get the nearest grid point to Alpha
            #print(list(set(T1['en'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) )])))
            #print(list(set(T1['en'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) )])))
            #print(list(set(T1['en'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] <= alpha) )])))
            #print(list(set(T1['en'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] <= alpha) )])))
            #print(list(set(T1['en'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] >= alpha) )])))
            #print(list(set(T1['en'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] >= alpha) )])))
            t0 = np.max(list(set(T1['en'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] <= alpha) )]) & 
                             set(T1['en'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] <= alpha) )])))
            t1 = np.min(list(set(T1['en'][np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] >= alpha) )]) & 
                             set(T1['en'][np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] >= alpha) )])))
            #print('alpha:', z0, alpha, z1)
    
    except:
        raise ValueError('Model Parameters Teff: %0.3f, logg: %0.3f, [M/H]: %0.3f, Alpha: %0.3f, kzz: %0.3f, CO: %0.3f are outside the model grid.'%(teff, logg, metal, alpha, kzz, co))


    if 'sonora' in modelset.lower() and '2023' not in modelset:

        # Get the 16 points
        ind0000 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0000
        ind0001 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0001
        ind0010 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0010
        ind0011 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0011
        ind0100 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0100
        ind0101 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0101
        ind0110 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0110
        ind0111 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0111
        ind1000 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1000
        ind1001 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1001
        ind1010 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1010
        ind1011 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1011
        ind1100 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1100
        ind1101 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1101
        ind1110 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1110
        ind1111 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1111
        Points =  [ [np.log10(T1['teff'][ind0000].data[0]), T1['logg'][ind0000].data[0], T1['FeH'][ind0000].data[0], T1['Y'][ind0000].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0000], logg=T1['logg'][ind0000], metal=T1['FeH'][ind0000], alpha=T1['Y'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0001].data[0]), T1['logg'][ind0001].data[0], T1['FeH'][ind0001].data[0], T1['Y'][ind0001].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0001], logg=T1['logg'][ind0001], metal=T1['FeH'][ind0001], alpha=T1['Y'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0010].data[0]), T1['logg'][ind0010].data[0], T1['FeH'][ind0010].data[0], T1['Y'][ind0010].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0010], logg=T1['logg'][ind0010], metal=T1['FeH'][ind0010], alpha=T1['Y'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0011].data[0]), T1['logg'][ind0011].data[0], T1['FeH'][ind0011].data[0], T1['Y'][ind0011].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0011], logg=T1['logg'][ind0011], metal=T1['FeH'][ind0011], alpha=T1['Y'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0100].data[0]), T1['logg'][ind0100].data[0], T1['FeH'][ind0100].data[0], T1['Y'][ind0100].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0100], logg=T1['logg'][ind0100], metal=T1['FeH'][ind0100], alpha=T1['Y'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0101].data[0]), T1['logg'][ind0101].data[0], T1['FeH'][ind0101].data[0], T1['Y'][ind0101].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0101], logg=T1['logg'][ind0101], metal=T1['FeH'][ind0101], alpha=T1['Y'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0110].data[0]), T1['logg'][ind0110].data[0], T1['FeH'][ind0110].data[0], T1['Y'][ind0110].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0110], logg=T1['logg'][ind0110], metal=T1['FeH'][ind0110], alpha=T1['Y'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0111].data[0]), T1['logg'][ind0111].data[0], T1['FeH'][ind0111].data[0], T1['Y'][ind0111].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0111], logg=T1['logg'][ind0111], metal=T1['FeH'][ind0111], alpha=T1['Y'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1000].data[0]), T1['logg'][ind1000].data[0], T1['FeH'][ind1000].data[0], T1['Y'][ind1000].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1000], logg=T1['logg'][ind1000], metal=T1['FeH'][ind1000], alpha=T1['Y'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1001].data[0]), T1['logg'][ind1001].data[0], T1['FeH'][ind1001].data[0], T1['Y'][ind1001].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1001], logg=T1['logg'][ind1001], metal=T1['FeH'][ind1001], alpha=T1['Y'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1010].data[0]), T1['logg'][ind1010].data[0], T1['FeH'][ind1010].data[0], T1['Y'][ind1010].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1010], logg=T1['logg'][ind1010], metal=T1['FeH'][ind1010], alpha=T1['Y'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1011].data[0]), T1['logg'][ind1011].data[0], T1['FeH'][ind1011].data[0], T1['Y'][ind1011].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1011], logg=T1['logg'][ind1011], metal=T1['FeH'][ind1011], alpha=T1['Y'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1100].data[0]), T1['logg'][ind1100].data[0], T1['FeH'][ind1100].data[0], T1['Y'][ind1100].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1100], logg=T1['logg'][ind1100], metal=T1['FeH'][ind1100], alpha=T1['Y'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1101].data[0]), T1['logg'][ind1101].data[0], T1['FeH'][ind1101].data[0], T1['Y'][ind1101].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1101], logg=T1['logg'][ind1101], metal=T1['FeH'][ind1101], alpha=T1['Y'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1110].data[0]), T1['logg'][ind1110].data[0], T1['FeH'][ind1110].data[0], T1['Y'][ind1110].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1110], logg=T1['logg'][ind1110], metal=T1['FeH'][ind1110], alpha=T1['Y'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1111].data[0]), T1['logg'][ind1111].data[0], T1['FeH'][ind1111].data[0], T1['Y'][ind1111].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)
    
    elif  modelset.lower() == 'sonora-2023':

        # Get the 32 points
        ind00000 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 00000
        ind00001 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 00001
        ind00010 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 00010
        ind00011 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 00011
        ind00100 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 00100
        ind00101 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 00101
        ind00110 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 00110
        ind00111 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 00111
        ind01000 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 01000
        ind01001 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 01001
        ind01010 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 01010
        ind01011 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 01011
        ind01100 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 01100
        ind01101 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 01101
        ind01110 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 01110
        ind01111 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 01111
        ind10000 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 10000
        ind10001 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 10001
        ind10010 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 10010
        ind10011 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 10011
        ind10100 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 10100
        ind10101 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 10101
        ind10110 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 10110
        ind10111 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 10111
        ind11000 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 11000
        ind11001 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 11001
        ind11010 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 11010
        ind11011 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 11011
        ind11100 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w0) ) # 11100
        ind11101 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t0) & (T1['CO'] == w1) ) # 11101
        ind11110 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w0) ) # 11110
        ind11111 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['kzz'] == t1) & (T1['CO'] == w1) ) # 11111
        Points =  [ [np.log10(T1['teff'][ind00000].data[0]), T1['logg'][ind00000].data[0], T1['M_H'][ind00000].data[0], np.log10(T1['kzz'][ind00000].data[0]), T1['CO'][ind00000].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00000], logg=T1['logg'][ind00000], metal=T1['M_H'][ind00000], kzz=T1['kzz'][ind00000], co=T1['CO'][ind00000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00001].data[0]), T1['logg'][ind00001].data[0], T1['M_H'][ind00001].data[0], np.log10(T1['kzz'][ind00001].data[0]), T1['CO'][ind00001].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00001], logg=T1['logg'][ind00001], metal=T1['M_H'][ind00001], kzz=T1['kzz'][ind00001], co=T1['CO'][ind00001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00010].data[0]), T1['logg'][ind00010].data[0], T1['M_H'][ind00010].data[0], np.log10(T1['kzz'][ind00010].data[0]), T1['CO'][ind00010].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00010], logg=T1['logg'][ind00010], metal=T1['M_H'][ind00010], kzz=T1['kzz'][ind00010], co=T1['CO'][ind00010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00011].data[0]), T1['logg'][ind00011].data[0], T1['M_H'][ind00011].data[0], np.log10(T1['kzz'][ind00011].data[0]), T1['CO'][ind00011].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00011], logg=T1['logg'][ind00011], metal=T1['M_H'][ind00011], kzz=T1['kzz'][ind00011], co=T1['CO'][ind00011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00100].data[0]), T1['logg'][ind00100].data[0], T1['M_H'][ind00100].data[0], np.log10(T1['kzz'][ind00100].data[0]), T1['CO'][ind00100].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00100], logg=T1['logg'][ind00100], metal=T1['M_H'][ind00100], kzz=T1['kzz'][ind00100], co=T1['CO'][ind00100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00101].data[0]), T1['logg'][ind00101].data[0], T1['M_H'][ind00101].data[0], np.log10(T1['kzz'][ind00101].data[0]), T1['CO'][ind00101].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00101], logg=T1['logg'][ind00101], metal=T1['M_H'][ind00101], kzz=T1['kzz'][ind00101], co=T1['CO'][ind00101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00110].data[0]), T1['logg'][ind00110].data[0], T1['M_H'][ind00110].data[0], np.log10(T1['kzz'][ind00110].data[0]), T1['CO'][ind00110].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00110], logg=T1['logg'][ind00110], metal=T1['M_H'][ind00110], kzz=T1['kzz'][ind00110], co=T1['CO'][ind00110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind00111].data[0]), T1['logg'][ind00111].data[0], T1['M_H'][ind00111].data[0], np.log10(T1['kzz'][ind00111].data[0]), T1['CO'][ind00111].data[0],                      
                     np.log10(GetModel(T1['teff'][ind00111], logg=T1['logg'][ind00111], metal=T1['M_H'][ind00111], kzz=T1['kzz'][ind00111], co=T1['CO'][ind00111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01000].data[0]), T1['logg'][ind01000].data[0], T1['M_H'][ind01000].data[0], np.log10(T1['kzz'][ind01000].data[0]), T1['CO'][ind01000].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01000], logg=T1['logg'][ind01000], metal=T1['M_H'][ind01000], kzz=T1['kzz'][ind01000], co=T1['CO'][ind01000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01001].data[0]), T1['logg'][ind01001].data[0], T1['M_H'][ind01001].data[0], np.log10(T1['kzz'][ind01001].data[0]), T1['CO'][ind01001].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01001], logg=T1['logg'][ind01001], metal=T1['M_H'][ind01001], kzz=T1['kzz'][ind01001], co=T1['CO'][ind01001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01010].data[0]), T1['logg'][ind01010].data[0], T1['M_H'][ind01010].data[0], np.log10(T1['kzz'][ind01010].data[0]), T1['CO'][ind01010].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01010], logg=T1['logg'][ind01010], metal=T1['M_H'][ind01010], kzz=T1['kzz'][ind01010], co=T1['CO'][ind01010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01011].data[0]), T1['logg'][ind01011].data[0], T1['M_H'][ind01011].data[0], np.log10(T1['kzz'][ind01011].data[0]), T1['CO'][ind01011].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01011], logg=T1['logg'][ind01011], metal=T1['M_H'][ind01011], kzz=T1['kzz'][ind01011], co=T1['CO'][ind01011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01100].data[0]), T1['logg'][ind01100].data[0], T1['M_H'][ind01100].data[0], np.log10(T1['kzz'][ind01100].data[0]), T1['CO'][ind01100].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01100], logg=T1['logg'][ind01100], metal=T1['M_H'][ind01100], kzz=T1['kzz'][ind01100], co=T1['CO'][ind01100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01101].data[0]), T1['logg'][ind01101].data[0], T1['M_H'][ind01101].data[0], np.log10(T1['kzz'][ind01101].data[0]), T1['CO'][ind01101].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01101], logg=T1['logg'][ind01101], metal=T1['M_H'][ind01101], kzz=T1['kzz'][ind01101], co=T1['CO'][ind01101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01110].data[0]), T1['logg'][ind01110].data[0], T1['M_H'][ind01110].data[0], np.log10(T1['kzz'][ind01110].data[0]), T1['CO'][ind01110].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01110], logg=T1['logg'][ind01110], metal=T1['M_H'][ind01110], kzz=T1['kzz'][ind01110], co=T1['CO'][ind01110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind01111].data[0]), T1['logg'][ind01111].data[0], T1['M_H'][ind01111].data[0], np.log10(T1['kzz'][ind01111].data[0]), T1['CO'][ind01111].data[0],                      
                     np.log10(GetModel(T1['teff'][ind01111], logg=T1['logg'][ind01111], metal=T1['M_H'][ind01111], kzz=T1['kzz'][ind01111], co=T1['CO'][ind01111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10000].data[0]), T1['logg'][ind10000].data[0], T1['M_H'][ind10000].data[0], np.log10(T1['kzz'][ind10000].data[0]), T1['CO'][ind10000].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10000], logg=T1['logg'][ind10000], metal=T1['M_H'][ind10000], kzz=T1['kzz'][ind10000], co=T1['CO'][ind10000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10001].data[0]), T1['logg'][ind10001].data[0], T1['M_H'][ind10001].data[0], np.log10(T1['kzz'][ind10001].data[0]), T1['CO'][ind10001].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10001], logg=T1['logg'][ind10001], metal=T1['M_H'][ind10001], kzz=T1['kzz'][ind10001], co=T1['CO'][ind10001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10010].data[0]), T1['logg'][ind10010].data[0], T1['M_H'][ind10010].data[0], np.log10(T1['kzz'][ind10010].data[0]), T1['CO'][ind10010].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10010], logg=T1['logg'][ind10010], metal=T1['M_H'][ind10010], kzz=T1['kzz'][ind10010], co=T1['CO'][ind10010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10011].data[0]), T1['logg'][ind10011].data[0], T1['M_H'][ind10011].data[0], np.log10(T1['kzz'][ind10011].data[0]), T1['CO'][ind10011].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10011], logg=T1['logg'][ind10011], metal=T1['M_H'][ind10011], kzz=T1['kzz'][ind10011], co=T1['CO'][ind10011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10100].data[0]), T1['logg'][ind10100].data[0], T1['M_H'][ind10100].data[0], np.log10(T1['kzz'][ind10100].data[0]), T1['CO'][ind10100].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10100], logg=T1['logg'][ind10100], metal=T1['M_H'][ind10100], kzz=T1['kzz'][ind10100], co=T1['CO'][ind10100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10101].data[0]), T1['logg'][ind10101].data[0], T1['M_H'][ind10101].data[0], np.log10(T1['kzz'][ind10101].data[0]), T1['CO'][ind10101].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10101], logg=T1['logg'][ind10101], metal=T1['M_H'][ind10101], kzz=T1['kzz'][ind10101], co=T1['CO'][ind10101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10110].data[0]), T1['logg'][ind10110].data[0], T1['M_H'][ind10110].data[0], np.log10(T1['kzz'][ind10110].data[0]), T1['CO'][ind10110].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10110], logg=T1['logg'][ind10110], metal=T1['M_H'][ind10110], kzz=T1['kzz'][ind10110], co=T1['CO'][ind10110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind10111].data[0]), T1['logg'][ind10111].data[0], T1['M_H'][ind10111].data[0], np.log10(T1['kzz'][ind10111].data[0]), T1['CO'][ind10111].data[0],                      
                     np.log10(GetModel(T1['teff'][ind10111], logg=T1['logg'][ind10111], metal=T1['M_H'][ind10111], kzz=T1['kzz'][ind10111], co=T1['CO'][ind10111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11000].data[0]), T1['logg'][ind11000].data[0], T1['M_H'][ind11000].data[0], np.log10(T1['kzz'][ind11000].data[0]), T1['CO'][ind11000].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11000], logg=T1['logg'][ind11000], metal=T1['M_H'][ind11000], kzz=T1['kzz'][ind11000], co=T1['CO'][ind11000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11001].data[0]), T1['logg'][ind11001].data[0], T1['M_H'][ind11001].data[0], np.log10(T1['kzz'][ind11001].data[0]), T1['CO'][ind11001].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11001], logg=T1['logg'][ind11001], metal=T1['M_H'][ind11001], kzz=T1['kzz'][ind11001], co=T1['CO'][ind11001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11010].data[0]), T1['logg'][ind11010].data[0], T1['M_H'][ind11010].data[0], np.log10(T1['kzz'][ind11010].data[0]), T1['CO'][ind11010].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11010], logg=T1['logg'][ind11010], metal=T1['M_H'][ind11010], kzz=T1['kzz'][ind11010], co=T1['CO'][ind11010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11011].data[0]), T1['logg'][ind11011].data[0], T1['M_H'][ind11011].data[0], np.log10(T1['kzz'][ind11011].data[0]), T1['CO'][ind11011].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11011], logg=T1['logg'][ind11011], metal=T1['M_H'][ind11011], kzz=T1['kzz'][ind11011], co=T1['CO'][ind11011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11100].data[0]), T1['logg'][ind11100].data[0], T1['M_H'][ind11100].data[0], np.log10(T1['kzz'][ind11100].data[0]), T1['CO'][ind11100].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11100], logg=T1['logg'][ind11100], metal=T1['M_H'][ind11100], kzz=T1['kzz'][ind11100], co=T1['CO'][ind11100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11101].data[0]), T1['logg'][ind11101].data[0], T1['M_H'][ind11101].data[0], np.log10(T1['kzz'][ind11101].data[0]), T1['CO'][ind11101].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11101], logg=T1['logg'][ind11101], metal=T1['M_H'][ind11101], kzz=T1['kzz'][ind11101], co=T1['CO'][ind11101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11110].data[0]), T1['logg'][ind11110].data[0], T1['M_H'][ind11110].data[0], np.log10(T1['kzz'][ind11110].data[0]), T1['CO'][ind11110].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11110], logg=T1['logg'][ind11110], metal=T1['M_H'][ind11110], kzz=T1['kzz'][ind11110], co=T1['CO'][ind11110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind11111].data[0]), T1['logg'][ind11111].data[0], T1['M_H'][ind11111].data[0], np.log10(T1['kzz'][ind11111].data[0]), T1['CO'][ind11111].data[0],                      
                     np.log10(GetModel(T1['teff'][ind11111], logg=T1['logg'][ind11111], metal=T1['M_H'][ind11111], kzz=T1['kzz'][ind11111], co=T1['CO'][ind11111], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['teff'][ind11111], logg=T1['logg'][ind11111], metal=T1['M_H'][ind11111], kzz=T1['kzz'][ind11111], co=T1['CO'][ind11111], instrument=instrument, order=order, gridfile=T1, wave=True)

    else:

        # Get the 16 points
        ind0000 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 0000
        ind0001 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 0001
        ind0010 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 0010
        ind0011 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 0011
        ind0100 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 0100
        ind0101 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 0101
        ind0110 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 0110
        ind0111 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 0111
        ind1000 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 1000
        ind1001 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 1001
        ind1010 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 1010
        ind1011 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 1011
        ind1100 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 1100
        ind1101 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 1101
        ind1110 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 1110
        ind1111 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 1111
        Points =  [ [np.log10(T1['teff'][ind0000].data[0]), T1['logg'][ind0000].data[0], T1['M_H'][ind0000].data[0], T1['en'][ind0000].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0000], logg=T1['logg'][ind0000], metal=T1['M_H'][ind0000], alpha=T1['en'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0001].data[0]), T1['logg'][ind0001].data[0], T1['M_H'][ind0001].data[0], T1['en'][ind0001].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0001], logg=T1['logg'][ind0001], metal=T1['M_H'][ind0001], alpha=T1['en'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0010].data[0]), T1['logg'][ind0010].data[0], T1['M_H'][ind0010].data[0], T1['en'][ind0010].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0010], logg=T1['logg'][ind0010], metal=T1['M_H'][ind0010], alpha=T1['en'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0011].data[0]), T1['logg'][ind0011].data[0], T1['M_H'][ind0011].data[0], T1['en'][ind0011].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0011], logg=T1['logg'][ind0011], metal=T1['M_H'][ind0011], alpha=T1['en'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0100].data[0]), T1['logg'][ind0100].data[0], T1['M_H'][ind0100].data[0], T1['en'][ind0100].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0100], logg=T1['logg'][ind0100], metal=T1['M_H'][ind0100], alpha=T1['en'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0101].data[0]), T1['logg'][ind0101].data[0], T1['M_H'][ind0101].data[0], T1['en'][ind0101].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0101], logg=T1['logg'][ind0101], metal=T1['M_H'][ind0101], alpha=T1['en'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0110].data[0]), T1['logg'][ind0110].data[0], T1['M_H'][ind0110].data[0], T1['en'][ind0110].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0110], logg=T1['logg'][ind0110], metal=T1['M_H'][ind0110], alpha=T1['en'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0111].data[0]), T1['logg'][ind0111].data[0], T1['M_H'][ind0111].data[0], T1['en'][ind0111].data[0],                          
                     np.log10(GetModel(T1['teff'][ind0111], logg=T1['logg'][ind0111], metal=T1['M_H'][ind0111], alpha=T1['en'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1000].data[0]), T1['logg'][ind1000].data[0], T1['M_H'][ind1000].data[0], T1['en'][ind1000].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1000], logg=T1['logg'][ind1000], metal=T1['M_H'][ind1000], alpha=T1['en'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1001].data[0]), T1['logg'][ind1001].data[0], T1['M_H'][ind1001].data[0], T1['en'][ind1001].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1001], logg=T1['logg'][ind1001], metal=T1['M_H'][ind1001], alpha=T1['en'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1010].data[0]), T1['logg'][ind1010].data[0], T1['M_H'][ind1010].data[0], T1['en'][ind1010].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1010], logg=T1['logg'][ind1010], metal=T1['M_H'][ind1010], alpha=T1['en'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1011].data[0]), T1['logg'][ind1011].data[0], T1['M_H'][ind1011].data[0], T1['en'][ind1011].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1011], logg=T1['logg'][ind1011], metal=T1['M_H'][ind1011], alpha=T1['en'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1100].data[0]), T1['logg'][ind1100].data[0], T1['M_H'][ind1100].data[0], T1['en'][ind1100].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1100], logg=T1['logg'][ind1100], metal=T1['M_H'][ind1100], alpha=T1['en'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1101].data[0]), T1['logg'][ind1101].data[0], T1['M_H'][ind1101].data[0], T1['en'][ind1101].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1101], logg=T1['logg'][ind1101], metal=T1['M_H'][ind1101], alpha=T1['en'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1110].data[0]), T1['logg'][ind1110].data[0], T1['M_H'][ind1110].data[0], T1['en'][ind1110].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1110], logg=T1['logg'][ind1110], metal=T1['M_H'][ind1110], alpha=T1['en'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1111].data[0]), T1['logg'][ind1111].data[0], T1['M_H'][ind1111].data[0], T1['en'][ind1111].data[0],                          
                     np.log10(GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['M_H'][ind1111], alpha=T1['en'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['M_H'][ind1111], alpha=T1['en'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)

    if modelset.lower() == 'sonora-2023':
        return waves2, smart.utils.interpolations.quintilinear_interpolation(np.log10(teff), logg, metal, np.log10(kzz), co, Points)
    else:
        return waves2, smart.utils.interpolations.quadlinear_interpolation(np.log10(teff), logg, metal, alpha, Points)

################################################################################################################################################################################################