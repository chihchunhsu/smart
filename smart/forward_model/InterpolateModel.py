import smart
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from astropy.io import fits
from numpy.linalg import inv, det
from ..utils.interpolations import trilinear_interpolation

##############################################################################################################


def InterpModel(teff, logg=4, metal=0, alpha=0, modelset='marcs-apogee-dr15', instrument='nirspec', order=33):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set and instrument
    if instrument == 'nirspec':
        if modelset.lower() == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'
        elif modelset.lower() == 'phoenix-aces-agss-cond-2011':
            path     = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams.csv'
        elif modelset.lower() == 'sonora-2018':
            path     = BASE + '/../libraries/SONORA_2018/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/SONORA_2018/SONORA_2018_gridparams.csv'

    elif instrument == 'apogee':
        if modelset.lower() == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'
        elif modelset.lower() == 'phoenix-btsettl-cifist2011-2015':
            path     = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/PHOENIX_BTSETTL_CIFIST2011_2015_gridparams_apogee.csv'
        elif modelset.lower() == 'phoenix-aces-agss-cond-2011' :
            #path     = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-RAW/'
            path     = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-ALL/'
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams_apogee.csv'
        elif modelset.lower() == 'marcs-apogee-dr15' :
            path     = BASE + '/../libraries/MARCS_APOGEE_DR15/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/MARCS_APOGEE_DR15/MARCS_APOGEE_DR15_gridparams_apogee.csv'

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

        if gridfile is None:
            raise ValueError('Model gridfile must be provided.') 

        if instrument == 'nirspec': 
            if modelset.lower() == 'btsettl08': 
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            elif modelset.lower() == 'phoenix-aces-agss-cond-2011':
                filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.fits'
            elif modelset.lower() == 'sonora-2018':
                filename = 'SONORA_2018_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_FeH{0:.2f}'.format(0) + '_Y{0:.2f}'.format(0.28) + '_CO{0:.2f}'.format(1.00) + '_NIRSPEC-O' + str(order) + '-RAW.txt'

        if instrument == 'apogee':
            filename = gridfile['File'][np.where( (gridfile['Temp']==temp) & (gridfile['Logg']==logg) & (gridfile['Metal']==metal) & (gridfile['Alpha']==alpha) )].data[0]

        if modelset.lower() == 'phoenix-aces-agss-cond-2011': Tab = Table.read(path+filename)
        else: Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    ###################################################################################

    # Check if the model already exists (grid point)
    if modelset.lower() == 'sonora-2018':
        if (teff, logg) in zip(T1['Temp'], T1['Logg']):
            metal, ys = 0, 0.28
            index0 = np.where( (T1['Temp'] == teff) & (T1['Logg'] == logg) & (T1['FeH'] == metal) & (T1['Y'] == ys) )
            #flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
            #waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['FeH'][index0], alpha=T1['Y'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
            return waves2, flux2
    else:
        if (teff, logg, metal, alpha) in zip(T1['Temp'], T1['Logg'], T1['Metal'], T1['Alpha']): 
            index0 = np.where( (T1['Temp'] == teff) & (T1['Logg'] == logg) & (T1['Metal'] == metal) & (T1['Alpha'] == alpha) )
            #flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
            #waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
            flux2  = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['Metal'][index0], alpha=T1['Alpha'][index0], instrument=instrument, order=order, gridfile=T1)
            waves2 = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['Metal'][index0], alpha=T1['Alpha'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
            return waves2, flux2


    try:
        if modelset.lower() == 'sonora-2018':
            metal, alpha = 0, 0.28
            # Get the nearest models to the gridpoint (Temp)
            x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= teff)])
            x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= teff)])
            #print(x0, x1)
            
            # Get the nearest grid point to Logg
            y0 = np.max(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] <= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] <= logg) )])))
            y1 = np.min(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] >= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] >= logg) )])))
            #print(y0, y1)
            
            # Get the nearest grid point to [M/H]
            z0 = np.max(list(set(T1['FeH'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] <= metal) )]) & 
                             set(T1['FeH'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] <= metal) )])))
            z1 = np.min(list(set(T1['FeH'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] >= metal) )]) & 
                             set(T1['FeH'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] >= metal) )])))
            #print(z0, z1)
            
            # Get the nearest grid point to Alpha
            t0 = np.max(list(set(T1['Y'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] <= alpha) )]) & 
                             set(T1['Y'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] <= alpha) )])))
            t1 = np.min(list(set(T1['Y'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] >= alpha) )]) & 
                             set(T1['Y'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] >= alpha) )])))
            #print(t0, t1)
            
        else:
            # Get the nearest models to the gridpoint (Temp)
            x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= teff)])
            x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= teff)])
            #print('teff:', x0, teff, x1)
            # Get the nearest grid point to Logg
            y0 = np.max(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] <= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] <= logg) )])))
            y1 = np.min(list(set(T1['Logg'][np.where( (T1['Temp'] == x0) & (T1['Logg'] >= logg) )]) & 
                             set(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] >= logg) )])))
            #print('logg:', y0, logg, y1)
            # Get the nearest grid point to [M/H]
            #print(metal)
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) )])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) )])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] <= metal))])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= metal))])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] >= metal))])))
            #print(list(set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= metal))])))
            z0 = np.max(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] <= metal) )]) & 
                             set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= metal) )])))
            z1 = np.min(list(set(T1['Metal'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] >= metal) )]) & 
                             set(T1['Metal'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= metal) )])))
            #print('metal:', z0, metal, z1)
            # Get the nearest grid point to Alpha
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] <= alpha) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] <= alpha) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] >= alpha) )])))
            #print(list(set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] >= alpha) )])))
            t0 = np.max(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] <= alpha) )]) & 
                             set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] <= alpha) )])))
            t1 = np.min(list(set(T1['Alpha'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] >= alpha) )]) & 
                             set(T1['Alpha'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] >= alpha) )])))
            #print('alpha:', z0, alpha, z1)
    except:
        raise ValueError('Model Parameters Teff: %0.3f, Logg: %0.3f, [M/H]: %0.3f, Alpha: %0.3f are outside the model grid.'%(teff, logg, metal, alpha))


    if modelset.lower() == 'sonora-2018':
        # Get the 16 points
        ind0000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0000
        ind1000 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1000
        ind0100 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 0100
        ind0010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0010
        ind0001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0001
        ind1001 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1001
        ind0101 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 0101
        ind0011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0011
        ind1011 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1011
        ind0111 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 0111
        ind1111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t1) ) # 1111
        ind0110 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 0110
        ind1010 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1010
        ind1100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t0) ) # 1100
        ind1101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z0) & (T1['Y'] == t1) ) # 1101
        ind1110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['FeH'] == z1) & (T1['Y'] == t0) ) # 1110
        Points =  [ [np.log10(T1['Temp'][ind0000]), T1['Logg'][ind0000], T1['FeH'][ind0000], T1['Y'][ind0000], 
                     np.log10(GetModel(T1['Temp'][ind0000], logg=T1['Logg'][ind0000], metal=T1['FeH'][ind0000], alpha=T1['Y'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1000]), T1['Logg'][ind1000], T1['FeH'][ind1000], T1['Y'][ind1000], 
                     np.log10(GetModel(T1['Temp'][ind1000], logg=T1['Logg'][ind1000], metal=T1['FeH'][ind1000], alpha=T1['Y'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0100]), T1['Logg'][ind0100], T1['FeH'][ind0100], T1['Y'][ind0100], 
                     np.log10(GetModel(T1['Temp'][ind0100], logg=T1['Logg'][ind0100], metal=T1['FeH'][ind0100], alpha=T1['Y'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0010]), T1['Logg'][ind0010], T1['FeH'][ind0010], T1['Y'][ind0010], 
                     np.log10(GetModel(T1['Temp'][ind0010], logg=T1['Logg'][ind0010], metal=T1['FeH'][ind0010], alpha=T1['Y'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0001]), T1['Logg'][ind0001], T1['FeH'][ind0001], T1['Y'][ind0001], 
                     np.log10(GetModel(T1['Temp'][ind0001], logg=T1['Logg'][ind0001], metal=T1['FeH'][ind0001], alpha=T1['Y'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1001]), T1['Logg'][ind1001], T1['FeH'][ind1001], T1['Y'][ind1001], 
                     np.log10(GetModel(T1['Temp'][ind1001], logg=T1['Logg'][ind1001], metal=T1['FeH'][ind1001], alpha=T1['Y'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0101]), T1['Logg'][ind0101], T1['FeH'][ind0101], T1['Y'][ind0101], 
                     np.log10(GetModel(T1['Temp'][ind0101], logg=T1['Logg'][ind0101], metal=T1['FeH'][ind0101], alpha=T1['Y'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0011]), T1['Logg'][ind0011], T1['FeH'][ind0011], T1['Y'][ind0011], 
                     np.log10(GetModel(T1['Temp'][ind0011], logg=T1['Logg'][ind0011], metal=T1['FeH'][ind0011], alpha=T1['Y'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1011]), T1['Logg'][ind1011], T1['FeH'][ind1011], T1['Y'][ind1011], 
                     np.log10(GetModel(T1['Temp'][ind1011], logg=T1['Logg'][ind1011], metal=T1['FeH'][ind1011], alpha=T1['Y'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0111]), T1['Logg'][ind0111], T1['FeH'][ind0111], T1['Y'][ind0111], 
                     np.log10(GetModel(T1['Temp'][ind0111], logg=T1['Logg'][ind0111], metal=T1['FeH'][ind0111], alpha=T1['Y'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1111]), T1['Logg'][ind1111], T1['FeH'][ind1111], T1['Y'][ind1111], 
                     np.log10(GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0110]), T1['Logg'][ind0110], T1['FeH'][ind0110], T1['Y'][ind0110], 
                     np.log10(GetModel(T1['Temp'][ind0110], logg=T1['Logg'][ind0110], metal=T1['FeH'][ind0110], alpha=T1['Y'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1010]), T1['Logg'][ind1010], T1['FeH'][ind1010], T1['Y'][ind1010], 
                     np.log10(GetModel(T1['Temp'][ind1010], logg=T1['Logg'][ind1010], metal=T1['FeH'][ind1010], alpha=T1['Y'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1100]), T1['Logg'][ind1100], T1['FeH'][ind1100], T1['Y'][ind1100], 
                     np.log10(GetModel(T1['Temp'][ind1100], logg=T1['Logg'][ind1100], metal=T1['FeH'][ind1100], alpha=T1['Y'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1101]), T1['Logg'][ind1101], T1['FeH'][ind1101], T1['Y'][ind1101], 
                     np.log10(GetModel(T1['Temp'][ind1101], logg=T1['Logg'][ind1101], metal=T1['FeH'][ind1101], alpha=T1['Y'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1110]), T1['Logg'][ind1110], T1['FeH'][ind1110], T1['Y'][ind1110], 
                     np.log10(GetModel(T1['Temp'][ind1110], logg=T1['Logg'][ind1110], metal=T1['FeH'][ind1110], alpha=T1['Y'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['FeH'][ind1111], alpha=T1['Y'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)
    else:
        # Get the 16 points
        ind0000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 0000
        ind1000 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 1000
        ind0100 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 0100
        ind0010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 0010
        ind0001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 0001
        ind1001 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 1001
        ind0101 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 0101
        ind0011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 0011
        ind1011 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 1011
        ind0111 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 0111
        ind1111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t1) ) # 1111
        ind0110 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 0110
        ind1010 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 1010
        ind1100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t0) ) # 1100
        ind1101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == t1) ) # 1101
        ind1110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == t0) ) # 1110
        Points =  [ [np.log10(T1['Temp'][ind0000]), T1['Logg'][ind0000], T1['Metal'][ind0000], T1['Alpha'][ind0000], 
                     np.log10(GetModel(T1['Temp'][ind0000], logg=T1['Logg'][ind0000], metal=T1['Metal'][ind0000], alpha=T1['Alpha'][ind0000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1000]), T1['Logg'][ind1000], T1['Metal'][ind1000], T1['Alpha'][ind1000], 
                     np.log10(GetModel(T1['Temp'][ind1000], logg=T1['Logg'][ind1000], metal=T1['Metal'][ind1000], alpha=T1['Alpha'][ind1000], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0100]), T1['Logg'][ind0100], T1['Metal'][ind0100], T1['Alpha'][ind0100], 
                     np.log10(GetModel(T1['Temp'][ind0100], logg=T1['Logg'][ind0100], metal=T1['Metal'][ind0100], alpha=T1['Alpha'][ind0100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0010]), T1['Logg'][ind0010], T1['Metal'][ind0010], T1['Alpha'][ind0010], 
                     np.log10(GetModel(T1['Temp'][ind0010], logg=T1['Logg'][ind0010], metal=T1['Metal'][ind0010], alpha=T1['Alpha'][ind0010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0001]), T1['Logg'][ind0001], T1['Metal'][ind0001], T1['Alpha'][ind0001], 
                     np.log10(GetModel(T1['Temp'][ind0001], logg=T1['Logg'][ind0001], metal=T1['Metal'][ind0001], alpha=T1['Alpha'][ind0001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1001]), T1['Logg'][ind1001], T1['Metal'][ind1001], T1['Alpha'][ind1001], 
                     np.log10(GetModel(T1['Temp'][ind1001], logg=T1['Logg'][ind1001], metal=T1['Metal'][ind1001], alpha=T1['Alpha'][ind1001], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0101]), T1['Logg'][ind0101], T1['Metal'][ind0101], T1['Alpha'][ind0101], 
                     np.log10(GetModel(T1['Temp'][ind0101], logg=T1['Logg'][ind0101], metal=T1['Metal'][ind0101], alpha=T1['Alpha'][ind0101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0011]), T1['Logg'][ind0011], T1['Metal'][ind0011], T1['Alpha'][ind0011], 
                     np.log10(GetModel(T1['Temp'][ind0011], logg=T1['Logg'][ind0011], metal=T1['Metal'][ind0011], alpha=T1['Alpha'][ind0011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1011]), T1['Logg'][ind1011], T1['Metal'][ind1011], T1['Alpha'][ind1011], 
                     np.log10(GetModel(T1['Temp'][ind1011], logg=T1['Logg'][ind1011], metal=T1['Metal'][ind1011], alpha=T1['Alpha'][ind1011], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0111]), T1['Logg'][ind0111], T1['Metal'][ind0111], T1['Alpha'][ind0111], 
                     np.log10(GetModel(T1['Temp'][ind0111], logg=T1['Logg'][ind0111], metal=T1['Metal'][ind0111], alpha=T1['Alpha'][ind0111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1111]), T1['Logg'][ind1111], T1['Metal'][ind1111], T1['Alpha'][ind1111], 
                     np.log10(GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['Metal'][ind1111], alpha=T1['Alpha'][ind1111], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind0110]), T1['Logg'][ind0110], T1['Metal'][ind0110], T1['Alpha'][ind0110], 
                     np.log10(GetModel(T1['Temp'][ind0110], logg=T1['Logg'][ind0110], metal=T1['Metal'][ind0110], alpha=T1['Alpha'][ind0110], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1010]), T1['Logg'][ind1010], T1['Metal'][ind1010], T1['Alpha'][ind1010], 
                     np.log10(GetModel(T1['Temp'][ind1010], logg=T1['Logg'][ind1010], metal=T1['Metal'][ind1010], alpha=T1['Alpha'][ind1010], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1100]), T1['Logg'][ind1100], T1['Metal'][ind1100], T1['Alpha'][ind1100], 
                     np.log10(GetModel(T1['Temp'][ind1100], logg=T1['Logg'][ind1100], metal=T1['Metal'][ind1100], alpha=T1['Alpha'][ind1100], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1101]), T1['Logg'][ind1101], T1['Metal'][ind1101], T1['Alpha'][ind1101], 
                     np.log10(GetModel(T1['Temp'][ind1101], logg=T1['Logg'][ind1101], metal=T1['Metal'][ind1101], alpha=T1['Alpha'][ind1101], instrument=instrument, order=order, gridfile=T1))],
                    [np.log10(T1['Temp'][ind1110]), T1['Logg'][ind1110], T1['Metal'][ind1110], T1['Alpha'][ind1110], 
                     np.log10(GetModel(T1['Temp'][ind1110], logg=T1['Logg'][ind1110], metal=T1['Metal'][ind1110], alpha=T1['Alpha'][ind1110], instrument=instrument, order=order, gridfile=T1))],
                  ]
        #print(Points)
        waves2 = GetModel(T1['Temp'][ind1111], logg=T1['Logg'][ind1111], metal=T1['Metal'][ind1111], alpha=T1['Alpha'][ind1111], instrument=instrument, order=order, gridfile=T1, wave=True)

    return waves2, smart.utils.interpolations.quadlinear_interpolation(np.log10(teff), logg, metal, alpha, Points)

################################################################################################################################################################################################


def InterpModel_3D(Teff, Logg, Metal, modelset='marcs-apogee-dr15', instrument='nirspec', order=33):
    Alpha = 0.0
    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set
    if instrument == 'nirspec':

        if modelset.lower() == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order

        elif modelset.lower() == 'phoenixaces' :
            path = BASE + '/../libraries/phoenixaces/NIRSPEC-O%s-RAW/'%order

    elif instrument == 'apogee':

        if modelset.lower() == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/APOGEE-RAW/'

        elif modelset.lower() == 'phoenix-btsettl-cifist2011-2015':
            path = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/APOGEE-RAW/'
        
        elif modelset.lower() == 'phoenix-aces-agss-cond-2011' :
            #path = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-RAW/'
            path = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-ALL/'

        elif modelset.lower() == 'marcs-apogee-dr15' :
            path = BASE + '/../libraries/MARCS_APOGEE_DR15/APOGEE-RAW_3D/'

    def GetModel(temp, logg, metal, modelset='marcs-apogee-dr15', wave=False):
        en = 0.00
        if instrument == 'nirspec':

            if modelset.lower() == 'btsettl08':
                # alpha enhancement correction
                if metal == -0.5:
                    en = 0.2
                elif (metal == -1.0) or (metal == -1.5) or (metal == -2.0) or (metal == -2.5):
                    en = 0.4

                if metal == 0.0:
                    filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
                elif metal == 0.5:
                    filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
                else:
                    filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            
            elif modelset.lower() == 'phoenixaces':
                filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
        
        elif instrument == 'apogee':
            if modelset.lower() == 'btsettl08':
                # alpha enhancement correction
                if metal == -0.5:
                    en = 0.2
                elif (metal == -1.0) or (metal == -1.5) or (metal == -2.0) or (metal == -2.5):
                    en = 0.4

                if metal == 0.0:
                    filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
                elif metal > 0.0:
                    filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
                else:
                    filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
            
            elif modelset.lower() == 'phoenix-btsettl-cifist2011-2015':
                filename = 'PHOENIX_BTSETTL_CIFIST2011_2015_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

            elif modelset.lower() == 'phoenix-aces-agss-cond-2011':
                #filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
                if metal == 0.0:
                    filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-ALL.fits'
                else:
                    filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-ALL.fits'
            
            elif modelset.lower() == 'marcs-apogee-dr15':
                filename = 'MARCS_APOGEE_DR15_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

        # read fits files
        if modelset.lower() == 'phoenix-aces-agss-cond-2011' and instrument == 'apogee':
            with fits.open(path+filename) as hdul:
                if wave:
                    return hdul[1].data['wave']
                else:
                    return hdul[1].data['flux']

        # read txt files
        else:
            Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

            if wave:
                return Tab['wave']
            else:
                return Tab['flux']

    

    if instrument == 'nirspec':

        if modelset.lower() == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'

        elif modelset.lower() == 'phoenixaces':
            Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'

    elif instrument == 'apogee':

        if modelset.lower() == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'

        elif modelset.lower() == 'phoenix-btsettl-cifist2011-2015':
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/PHOENIX_BTSETTL_CIFIST2011_2015_gridparams_apogee.csv'

        elif modelset.lower() == 'phoenix-aces-agss-cond-2011':
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams_apogee.csv'

        elif modelset.lower() == 'marcs-apogee-dr15':
            Gridfile = BASE + '/../libraries/MARCS_APOGEE_DR15/MARCS_APOGEE_DR15_3D_gridparams_apogee.csv'

    T1 = Table.read(Gridfile)
    # Check if the model already exists (grid point)
    if (Teff, Logg, Metal) in zip(T1['Temp'], T1['Logg'], T1['Metal']):
        if modelset.lower() == 'btsettl08':
            if Metal == -0.5:
                Alpha = 0.2
            elif (Metal == -1.0) or (Metal == -1.5) or (Metal == -2.0) or (Metal == -2.5):
                Alpha = 0.4
        index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) & (T1['Metal'] == Metal) & (T1['Alpha'] == Alpha) )
        flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
        waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
        return waves2, flux2

    # Get the nearest models to the gridpoint (Temp)
    x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= Teff)])
    x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= Teff)])
    #print(x0, Teff, x1)
    #y0 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] <= Logg) )][-1]
    #y1 = T1['Logg'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & (T1['Logg'] >= Logg) )][0]
    #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )])))
    #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
    #print(x0, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )])))
    #print(x1, list(set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
    y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
    y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))
    #print(y0, Logg, y1)
    #z0 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS) )][-1]
    #z1 = T1['pgs'][np.where( ( (T1['Temp'] == x0) | (T1['Temp'] == x1) ) & ( (T1['Logg'] == y0) | (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS) )][0]
    #print(x0, y0, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['pgs'] <= PGS))])))
    #print(x1, y1, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['pgs'] <= PGS))])))
    #print(x0, y0, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['pgs'] >= PGS))])))
    #print(x1, y1, list(set(T1['pgs'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) ) & (T1['pgs'] >= PGS))])))
    z0 = np.max(list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] <= Metal) )]) & set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] <= Metal)) )])))
    z1 = np.min(list(set(T1['Metal'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] == y0) ) & (T1['Metal'] >= Metal) )]) & set(T1['Metal'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] >= Metal)) )])))
    #print(z0, PGS, z1)

    # Check if the gridpoint exists within the model ranges
    for x in [x0, x1]:
        for y in [y0, y1]:
            for z in [z0, z1]:
                if (x, y, z) not in zip(T1['Temp'], T1['Logg'], T1['Metal']):
                    print('No Model', x, y, z)
                    return 1
    '''
    print(np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1)))
    print(np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2)))
    print(np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1)))
    print(np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2)))
    print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]))
    print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))]))
    print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))]))
    print(np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))]), np.log10(T1['Logg'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))]))
    '''
    # Get the 16 points
    if modelset.lower() == 'btsettl08':
        if z0 == -0.5:
            Alpha0 = 0.2
        elif (z0 == -1.0) or (z0 == -1.5) or (z0 == -2.0) or (z0 == -2.5):
            Alpha0 = 0.4
        else:
            Alpha0 = 0.0
        if z1 == -0.5:
            Alpha1 = 0.2
        elif (z1 == -1.0) or (z1 == -1.5) or (z1 == -2.0) or (z1 == -2.5):
            Alpha1 = 0.4
        else:
            Alpha1 = 0.0
        ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha0) ) # 000
        ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha0) ) # 100
        ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha0) ) # 010
        ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha0) ) # 110
        ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha1) ) # 001
        ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha1) ) # 101
        ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha1) ) # 011
        ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha1) ) # 111

    else:
        ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha) ) # 000
        ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha) ) # 100
        ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha) ) # 010
        ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) & (T1['Alpha'] == Alpha) ) # 110
        ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha) ) # 001
        ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha) ) # 101
        ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha) ) # 011
        ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) & (T1['Alpha'] == Alpha) ) # 111

    Points =  [ [np.log10(T1['Temp'][ind000]), T1['Logg'][ind000], T1['Metal'][ind000], 
                 np.log10(GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['Metal'][ind000], modelset=modelset))],
                [np.log10(T1['Temp'][ind100]), T1['Logg'][ind100], T1['Metal'][ind100], 
                 np.log10(GetModel(T1['Temp'][ind100], T1['Logg'][ind100], T1['Metal'][ind100], modelset=modelset))],
                [np.log10(T1['Temp'][ind010]), T1['Logg'][ind010], T1['Metal'][ind010],  
                 np.log10(GetModel(T1['Temp'][ind010], T1['Logg'][ind010], T1['Metal'][ind010], modelset=modelset))],
                [np.log10(T1['Temp'][ind110]), T1['Logg'][ind110], T1['Metal'][ind110],  
                 np.log10(GetModel(T1['Temp'][ind110], T1['Logg'][ind110], T1['Metal'][ind110], modelset=modelset))],
                [np.log10(T1['Temp'][ind001]), T1['Logg'][ind001], T1['Metal'][ind001], 
                 np.log10(GetModel(T1['Temp'][ind001], T1['Logg'][ind001], T1['Metal'][ind001], modelset=modelset))],
                [np.log10(T1['Temp'][ind101]), T1['Logg'][ind101], T1['Metal'][ind101], 
                 np.log10(GetModel(T1['Temp'][ind101], T1['Logg'][ind101], T1['Metal'][ind101], modelset=modelset))],
                [np.log10(T1['Temp'][ind011]), T1['Logg'][ind011], T1['Metal'][ind011], 
                 np.log10(GetModel(T1['Temp'][ind011], T1['Logg'][ind011], T1['Metal'][ind011], modelset=modelset))],
                [np.log10(T1['Temp'][ind111]), T1['Logg'][ind111], T1['Metal'][ind111], 
                 np.log10(GetModel(T1['Temp'][ind111], T1['Logg'][ind111], T1['Metal'][ind111], modelset=modelset))],
              ]
    #print(Points)
    waves2 = GetModel(T1['Temp'][ind000], T1['Logg'][ind000], T1['Metal'][ind000], wave=True, modelset=modelset)

    return waves2, trilinear_interpolation(np.log10(Teff), Logg, Metal, Points)



    ################################################################################################
