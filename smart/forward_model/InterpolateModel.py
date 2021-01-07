import smart
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from astropy.io import fits
from numpy.linalg import inv, det
from ..utils.interpolations import trilinear_interpolation

##############################################################################################################


def InterpModel(teff, logg=4, metal=0, alpha=0, modelset='phoenix-aces-agss-cond-2011', instrument='nirspec', order='O33'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set and instrument
    if instrument.lower() == 'nirspec':
        path     = BASE + '/../libraries/%s/%s-O%s/'%(smart.ModelSets[modelset.lower()], instrument.upper(), order.upper())
    else:
        path     = BASE + '/../libraries/%s/%s-%s/'%(smart.ModelSets[modelset.lower()], instrument.upper(), order.upper())
    Gridfile = BASE + '/../libraries/%s/%s_gridparams.csv'%(smart.ModelSets[modelset.lower()], smart.ModelSets[modelset.lower()])

    if modelset.lower() == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'

    # Read the grid file
    T1 = Table.read(Gridfile)

    ###################################################################################

    def GetModel(temp, wave=False, **kwargs):
        
        logg       = kwargs.get('logg', 4.5)
        metal      = kwargs.get('metal', 0)
        alpha      = kwargs.get('alpha', 0)
        gridfile   = kwargs.get('gridfile', None)
        instrument = kwargs.get('instrument', 'nirspec')
        order      = kwargs.get('order', None)
        #print(temp, logg, metal, alpha)
        if gridfile is None:
            raise ValueError('Model gridfile must be provided.') 
        
        if modelset.lower() == 'btsettl08': 
            filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'    
        elif modelset.lower() == 'sonora':
            if instrument.lower() == 'nirspec':
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_FeH0.00_Y0.28_CO1.00' + '_%s-O%s.fits'%(instrument.upper(), order.upper())
            else:
                filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_FeH0.00_Y0.28_CO1.00' + '_%s-%s.fits'%(instrument.upper(), order.upper())
        elif modelset.lower() == 'marcs-apogee-dr15':
            cm      = kwargs.get('cm', 0)
            nm      = kwargs.get('nm', 0) 
            filename = '%s'%smart.ModelSets[modelset.lower()] + '_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_cm{0:.2f}'.format(float(cm)) + '_nm{0:.2f}'.format(float(nm)) + '_%s-%s.fits'%(instrument.upper(), order.upper())
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
    if modelset.lower() == 'sonora':
        if (teff, logg) in zip(T1['teff'], T1['logg']):
            metal, ys = 0, 0.28
            index0 = np.where( (T1['teff'] == teff) & (T1['logg'] == logg) & (T1['FeH'] == metal) & (T1['Y'] == ys) )
            #flux2  = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset )
            #waves2 = GetModel(T1['teff'][index0], T1['logg'][index0], T1['M_H'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['teff'][index0], logg=T1['logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
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
        if modelset.lower() == 'sonora':
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
        raise ValueError('Model Parameters Teff: %0.3f, logg: %0.3f, [M/H]: %0.3f, Alpha: %0.3f are outside the model grid.'%(teff, logg, metal, alpha))


    if modelset.lower() == 'sonora':
        # Get the 16 points
        ind0000 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0000
        ind1000 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1000
        ind0100 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0100
        ind0010 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0010
        ind0001 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0001
        ind1001 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1001
        ind0101 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0101
        ind0011 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0011
        ind1011 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1011
        ind0111 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0111
        ind1111 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1111
        ind0110 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0110
        ind1010 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1010
        ind1100 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1100
        ind1101 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1101
        ind1110 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1110
        Points =  [ [np.log10(T1['teff'][ind0000]), T1['logg'][ind0000], T1['FeH'][ind0000], T1['Y'][ind0000], 
                     np.log10(GetModel(T1['teff'][ind0000], logg=T1['logg'][ind0000], metal=T1['FeH'][ind0000], alpha=T1['Y'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1000]), T1['logg'][ind1000], T1['FeH'][ind1000], T1['Y'][ind1000], 
                     np.log10(GetModel(T1['teff'][ind1000], logg=T1['logg'][ind1000], metal=T1['FeH'][ind1000], alpha=T1['Y'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0100]), T1['logg'][ind0100], T1['FeH'][ind0100], T1['Y'][ind0100], 
                     np.log10(GetModel(T1['teff'][ind0100], logg=T1['logg'][ind0100], metal=T1['FeH'][ind0100], alpha=T1['Y'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0010]), T1['logg'][ind0010], T1['FeH'][ind0010], T1['Y'][ind0010], 
                     np.log10(GetModel(T1['teff'][ind0010], logg=T1['logg'][ind0010], metal=T1['FeH'][ind0010], alpha=T1['Y'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0001]), T1['logg'][ind0001], T1['FeH'][ind0001], T1['Y'][ind0001], 
                     np.log10(GetModel(T1['teff'][ind0001], logg=T1['logg'][ind0001], metal=T1['FeH'][ind0001], alpha=T1['Y'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1001]), T1['logg'][ind1001], T1['FeH'][ind1001], T1['Y'][ind1001], 
                     np.log10(GetModel(T1['teff'][ind1001], logg=T1['logg'][ind1001], metal=T1['FeH'][ind1001], alpha=T1['Y'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0101]), T1['logg'][ind0101], T1['FeH'][ind0101], T1['Y'][ind0101], 
                     np.log10(GetModel(T1['teff'][ind0101], logg=T1['logg'][ind0101], metal=T1['FeH'][ind0101], alpha=T1['Y'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0011]), T1['logg'][ind0011], T1['FeH'][ind0011], T1['Y'][ind0011], 
                     np.log10(GetModel(T1['teff'][ind0011], logg=T1['logg'][ind0011], metal=T1['FeH'][ind0011], alpha=T1['Y'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1011]), T1['logg'][ind1011], T1['FeH'][ind1011], T1['Y'][ind1011], 
                     np.log10(GetModel(T1['teff'][ind1011], logg=T1['logg'][ind1011], metal=T1['FeH'][ind1011], alpha=T1['Y'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0111]), T1['logg'][ind0111], T1['FeH'][ind0111], T1['Y'][ind0111], 
                     np.log10(GetModel(T1['teff'][ind0111], logg=T1['logg'][ind0111], metal=T1['FeH'][ind0111], alpha=T1['Y'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1111]), T1['logg'][ind1111], T1['FeH'][ind1111], T1['Y'][ind1111], 
                     np.log10(GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0110]), T1['logg'][ind0110], T1['FeH'][ind0110], T1['Y'][ind0110], 
                     np.log10(GetModel(T1['teff'][ind0110], logg=T1['logg'][ind0110], metal=T1['FeH'][ind0110], alpha=T1['Y'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1010]), T1['logg'][ind1010], T1['FeH'][ind1010], T1['Y'][ind1010], 
                     np.log10(GetModel(T1['teff'][ind1010], logg=T1['logg'][ind1010], metal=T1['FeH'][ind1010], alpha=T1['Y'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1100]), T1['logg'][ind1100], T1['FeH'][ind1100], T1['Y'][ind1100], 
                     np.log10(GetModel(T1['teff'][ind1100], logg=T1['logg'][ind1100], metal=T1['FeH'][ind1100], alpha=T1['Y'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1101]), T1['logg'][ind1101], T1['FeH'][ind1101], T1['Y'][ind1101], 
                     np.log10(GetModel(T1['teff'][ind1101], logg=T1['logg'][ind1101], metal=T1['FeH'][ind1101], alpha=T1['Y'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1110]), T1['logg'][ind1110], T1['FeH'][ind1110], T1['Y'][ind1110], 
                     np.log10(GetModel(T1['teff'][ind1110], logg=T1['logg'][ind1110], metal=T1['FeH'][ind1110], alpha=T1['Y'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)
    else:
        # Get the 16 points
        ind0000 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 0000
        ind1000 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 1000
        ind0100 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 0100
        ind0010 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 0010
        ind0001 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 0001
        ind1001 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 1001
        ind0101 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 0101
        ind0011 = np.where( (T1['teff'] == x0) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 0011
        ind1011 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 1011
        ind0111 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 0111
        ind1111 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t1) ) # 1111
        ind0110 = np.where( (T1['teff'] == x0) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 0110
        ind1010 = np.where( (T1['teff'] == x1) & (T1['logg'] == y0) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 1010
        ind1100 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t0) ) # 1100
        ind1101 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z0) & (T1['en'] == t1) ) # 1101
        ind1110 = np.where( (T1['teff'] == x1) & (T1['logg'] == y1) & (T1['M_H'] == z1) & (T1['en'] == t0) ) # 1110
        Points =  [ [np.log10(T1['teff'][ind0000]), T1['logg'][ind0000], T1['M_H'][ind0000], T1['en'][ind0000], 
                     np.log10(GetModel(T1['teff'][ind0000], logg=T1['logg'][ind0000], metal=T1['M_H'][ind0000], alpha=T1['en'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1000]), T1['logg'][ind1000], T1['M_H'][ind1000], T1['en'][ind1000], 
                     np.log10(GetModel(T1['teff'][ind1000], logg=T1['logg'][ind1000], metal=T1['M_H'][ind1000], alpha=T1['en'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0100]), T1['logg'][ind0100], T1['M_H'][ind0100], T1['en'][ind0100], 
                     np.log10(GetModel(T1['teff'][ind0100], logg=T1['logg'][ind0100], metal=T1['M_H'][ind0100], alpha=T1['en'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0010]), T1['logg'][ind0010], T1['M_H'][ind0010], T1['en'][ind0010], 
                     np.log10(GetModel(T1['teff'][ind0010], logg=T1['logg'][ind0010], metal=T1['M_H'][ind0010], alpha=T1['en'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0001]), T1['logg'][ind0001], T1['M_H'][ind0001], T1['en'][ind0001], 
                     np.log10(GetModel(T1['teff'][ind0001], logg=T1['logg'][ind0001], metal=T1['M_H'][ind0001], alpha=T1['en'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1001]), T1['logg'][ind1001], T1['M_H'][ind1001], T1['en'][ind1001], 
                     np.log10(GetModel(T1['teff'][ind1001], logg=T1['logg'][ind1001], metal=T1['M_H'][ind1001], alpha=T1['en'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0101]), T1['logg'][ind0101], T1['M_H'][ind0101], T1['en'][ind0101], 
                     np.log10(GetModel(T1['teff'][ind0101], logg=T1['logg'][ind0101], metal=T1['M_H'][ind0101], alpha=T1['en'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0011]), T1['logg'][ind0011], T1['M_H'][ind0011], T1['en'][ind0011], 
                     np.log10(GetModel(T1['teff'][ind0011], logg=T1['logg'][ind0011], metal=T1['M_H'][ind0011], alpha=T1['en'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1011]), T1['logg'][ind1011], T1['M_H'][ind1011], T1['en'][ind1011], 
                     np.log10(GetModel(T1['teff'][ind1011], logg=T1['logg'][ind1011], metal=T1['M_H'][ind1011], alpha=T1['en'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0111]), T1['logg'][ind0111], T1['M_H'][ind0111], T1['en'][ind0111], 
                     np.log10(GetModel(T1['teff'][ind0111], logg=T1['logg'][ind0111], metal=T1['M_H'][ind0111], alpha=T1['en'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1111]), T1['logg'][ind1111], T1['M_H'][ind1111], T1['en'][ind1111], 
                     np.log10(GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['M_H'][ind1111], alpha=T1['en'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind0110]), T1['logg'][ind0110], T1['M_H'][ind0110], T1['en'][ind0110], 
                     np.log10(GetModel(T1['teff'][ind0110], logg=T1['logg'][ind0110], metal=T1['M_H'][ind0110], alpha=T1['en'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1010]), T1['logg'][ind1010], T1['M_H'][ind1010], T1['en'][ind1010], 
                     np.log10(GetModel(T1['teff'][ind1010], logg=T1['logg'][ind1010], metal=T1['M_H'][ind1010], alpha=T1['en'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1100]), T1['logg'][ind1100], T1['M_H'][ind1100], T1['en'][ind1100], 
                     np.log10(GetModel(T1['teff'][ind1100], logg=T1['logg'][ind1100], metal=T1['M_H'][ind1100], alpha=T1['en'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1101]), T1['logg'][ind1101], T1['M_H'][ind1101], T1['en'][ind1101], 
                     np.log10(GetModel(T1['teff'][ind1101], logg=T1['logg'][ind1101], metal=T1['M_H'][ind1101], alpha=T1['en'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['teff'][ind1110]), T1['logg'][ind1110], T1['M_H'][ind1110], T1['en'][ind1110], 
                     np.log10(GetModel(T1['teff'][ind1110], logg=T1['logg'][ind1110], metal=T1['M_H'][ind1110], alpha=T1['en'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['teff'][ind1111], logg=T1['logg'][ind1111], metal=T1['M_H'][ind1111], alpha=T1['en'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)

    return waves2, smart.utils.interpolations.quadlinear_interpolation(np.log10(teff), logg, metal, alpha, Points)

################################################################################################################################################################################################
