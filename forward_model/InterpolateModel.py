import nirspec_fmp as nsp
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from numpy.linalg import inv, det


##############################################################################################################


def InterpModel(teff, logg=4, metal=0, alpha=0, modelset='marcs-apogee-dr15', instrument='nirspec', order=33):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set and instrument
    if instrument == 'nirspec':
        if modelset == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'
        elif modelset == 'phoenix-aces-agss-cond-2011':
            path     = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/NIRSPEC-O%s-RAW/'%order
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams.csv'

    elif instrument == 'apogee':
        if modelset == 'btsettl08':
            path     = BASE + '/../libraries/btsettl08/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'
        elif modelset == 'phoenix-btsettl-cifist2011-2015':
            path     = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/PHOENIX_BTSETTL_CIFIST2011_2015_gridparams_apogee.csv'
        elif modelset == 'phoenix-aces-agss-cond-2011' :
            path     = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-RAW/'
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams_apogee.csv'
        elif modelset == 'marcs-apogee-dr15' :
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
            if modelset == 'btsettl08': # NEED TO FIX THESE MODELS!
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            elif modelset == 'phoenix-aces-agss-cond-2011':
                filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_alpha{0:.2f}'.format(float(alpha)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
                #filename = gridfile['File'][np.where( (gridfile['Temp']==temp) & (gridfile['Logg']==logg) & (gridfile['Metal']==metal) & (gridfile['Alpha']==alpha) )].data[0]

        if instrument == 'apogee':
            filename = gridfile['File'][np.where( (gridfile['Temp']==temp) & (gridfile['Logg']==logg) & (gridfile['Metal']==metal) & (gridfile['Alpha']==alpha) )].data[0]

        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    ###################################################################################

    # Check if the model already exists (grid point)
    if (teff, logg, metal, alpha) in zip(T1['Temp'], T1['Logg'], T1['Metal'], T1['Alpha']): 
        index0 = np.where( (T1['Temp'] == teff) & (T1['Logg'] == logg) & (T1['Metal'] == metal) & (T1['Alpha'] == alpha) )
        #flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset )
        #waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], T1['Metal'][index0], modelset=modelset, wave=True)
        flux2  = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['Metal'][index0], alpha=T1['Alpha'][index0], instrument=instrument, order=order, gridfile=T1)
        waves2 = GetModel(T1['Temp'][index0], logg=T1['Logg'][index0], metal=T1['Metal'][index0], alpha=T1['Alpha'][index0], instrument=instrument, order=order, gridfile=T1, wave=True)
        return waves2, flux2


    try:
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

    return waves2, nsp.utils.interpolations.quadlinear_interpolation(np.log10(teff), logg, metal, alpha, Points)

