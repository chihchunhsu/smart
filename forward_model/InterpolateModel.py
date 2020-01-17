#!/usr/bin/env python
import numpy as np
import sys, os, os.path, time
from astropy.table import Table
from numpy.linalg import inv, det


################################################################

def InterpModel(Teff, Logg, modelset='btsettl08', order=33, instrument='nirspec'): # 2D interpolation

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set
    if instrument == 'nirspec':
        if modelset == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order
        elif modelset == 'phoenixaces' :
            path = BASE + '/../libraries/phoenixaces/NIRSPEC-O%s-RAW/'%order

    elif instrument == 'apogee':
        if modelset == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/APOGEE-RAW/'
        elif modelset == 'phoenix-btsettl-cifist2011-2015':
            path = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/APOGEE-RAW/'
        elif modelset == 'phoenix-aces-agss-cond-2011' :
            path = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-RAW/'
        elif modelset == 'marcs-apogee-dr15' :
            path = BASE + '/../libraries/MARCS_APOGEE_DR15/APOGEE-RAW/'

    def bilinear_interpolation(x, y, points):
        '''Interpolate (x,y) from values associated with four points.

        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.

            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        try:
            points = sorted(points)               # order points by x, then by y

            (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
    
            if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
                raise ValueError('points do not form a rectangle')
            if not x1 <= x <= x2 or not y1 <= y <= y2:
                raise ValueError('(x, y) not within the rectangle')
    
            return 10**((q11 * (x2 - x) * (y2 - y) +
                    q21 * (x - x1) * (y2 - y) +
                    q12 * (x2 - x) * (y - y1) +
                    q22 * (x - x1) * (y - y1)
                   ) / ((x2 - x1) * (y2 - y1) + 0.0))
        except:
            # handling linear interpolation, it does not matter which y1 or y2 is larger
            # see formula at: https://en.wikipedia.org/wiki/Linear_interpolation
            (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

            if x2 == x1 == _x1 == _x2: # Need to linearly interpolate along the y axis
                return 10**( ( q11 * ( y2 - y ) + q22 * ( y - y1 ) ) / ( ( y2 - y1 ) ) )

            if y2 == y1 == _y1 == _y2: # Need to linearly interpolate along the x axis
                return 10**( ( q11 * ( x2 - x ) + q22 * ( x - x1 ) ) / ( ( x2 - x1 ) ) )

            if x1 != _x1 or x2 != _x2:
                if x1 == x2 and _x1 == _x2:
                    x2  = _x1
                    q11 = q22

                    return 10**( ( q21 * ( x2 - x ) + q22 * ( x - x1 ) ) / ( ( x2 - x1 ) ) )

                else:
                    raise ValueError('points do not form a line') 

            if y1 != _y1 or y2 != _y2:
                if y1 == y2 and _y1 == _y2:
                    y2  = _y1
                    q12 = q21

                    return 10**( ( q11 * ( y2 - y ) + q12 * ( y - y1 ) ) / ( ( y2 - y1 ) ) )

                else:
                    raise ValueError('points do not form a line')      

    def GetModel(temp, logg, modelset='btsettl08', wave=False, instrument=instrument):
        feh, en = 0.00, 0.00
        if instrument == 'nirspec':

            if modelset == 'btsettl08':
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            
            elif modelset == 'phoenixaces':
                filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
        
        elif instrument == 'apogee':
            
            if modelset == 'btsettl08':
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
            
            elif modelset == 'phoenix-btsettl-cifist2011-2015':
                filename = 'PHOENIX_BTSETTL_CIFIST2011_2015_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

            elif modelset == 'phoenix-aces-agss-cond-2011':
                filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
            
            elif modelset == 'marcs-apogee-dr15':
                filename = 'MARCS_APOGEE_DR15_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']



    if instrument == 'nirspec':

        if modelset == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'

        elif modelset == 'phoenixaces':
            Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'

    elif instrument == 'apogee':

        if modelset == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'

        elif modelset == 'phoenix-btsettl-cifist2011-2015':
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/PHOENIX_BTSETTL_CIFIST2011_2015_gridparams_apogee.csv'

        elif modelset == 'phoenix-aces-agss-cond-2011':
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams_apogee.csv'

        elif modelset == 'marcs-apogee-dr15':
            Gridfile = BASE + '/../libraries/MARCS_APOGEE_DR15/MARCS_APOGEE_DR15_gridparams_apogee.csv'

    T1 = Table.read(Gridfile)

    # Check if the model already exists (grid point)
    if (Teff, Logg) in zip(T1['Temp'], T1['Logg']): 
        index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) )
        flux2  = GetModel(T1['Temp'][index0], T1['Logg'][index0], modelset=modelset )
        waves2 = GetModel(T1['Temp'][index0], T1['Logg'][index0], modelset=modelset, wave=True)
        return waves2, flux2

    x0 = np.max(T1['Temp'][np.where(T1['Temp'] <= Teff)])
    x1 = np.min(T1['Temp'][np.where(T1['Temp'] >= Teff)])
    y0 = np.max(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] <= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] <= Logg) ) )])))
    y1 = np.min(list(set(T1['Logg'][np.where( ( (T1['Temp'] == x0) & (T1['Logg'] >= Logg) ) )]) & set(T1['Logg'][np.where( ( (T1['Temp'] == x1) & (T1['Logg'] >= Logg) ) )])))

    # Check if the gridpoint exists within the model ranges
    for x in [x0, x1]:
        for y in [y0, y1]:
            if (x, y) not in zip(T1['Temp'], T1['Logg']):
                print('No Model', x, y)
                return 1

    print(Teff, Logg, x0, x1, y0, y1)
    
    # Get the four points
    Points =  [ [np.log10(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))]), T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y1))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y0))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y0))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], modelset=modelset))],
              ]

    waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0))], T1['Logg'][np.where((T1['Temp'] == x0) & (T1['Logg'] == y0))], wave=True, modelset=modelset)

    return waves2, bilinear_interpolation(np.log10(Teff), Logg, Points)


################################################################################################################################################################################################


def InterpModel_3D(Teff, Logg, Metal, modelset='marcs-apogee-dr15', instrument='nirspec', order=33):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set
    if instrument == 'nirspec':

        if modelset == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order

        elif modelset == 'phoenixaces' :
            path = BASE + '/../libraries/phoenixaces/NIRSPEC-O%s-RAW/'%order

    elif instrument == 'apogee':

        if modelset == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/APOGEE-RAW/'

        elif modelset == 'phoenix-btsettl-cifist2011-2015':
            path = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/APOGEE-RAW/'
        
        elif modelset == 'phoenix-aces-agss-cond-2011' :
            path = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/APOGEE-RAW/'

        elif modelset == 'marcs-apogee-dr15' :
            path = BASE + '/../libraries/MARCS_APOGEE_DR15/APOGEE-RAW_3D/'
        

    def trilinear_interpolation(x, y, z, points):
        '''Interpolate (x,y) from values associated with 9 points.

        Custom routine

        '''

        (x0, y0, z0, q000), (x1, y0, z0, q100), (x0, y1, z0, q010), (x1, y1, z0, q110), \
        (x0, y0, z1, q001), (x1, y0, z1, q101), (x0, y1, z1, q011), (x1, y1, z1, q111),  = points
        x0 = x0.data[0]
        x1 = x1.data[0]
        y0 = y0.data[0]
        y1 = y1.data[0]
        z0 = z0.data[0]
        z1 = z1.data[0]

        c = np.array([ [1., x0, y0, z0, x0*y0, x0*z0, y0*z0, x0*y0*z0], #000
                       [1., x1, y0, z0, x1*y0, x1*z0, y0*z0, x1*y0*z0], #100
                       [1., x0, y1, z0, x0*y1, x0*z0, y1*z0, x0*y1*z0], #010
                       [1., x1, y1, z0, x1*y1, x1*z0, y1*z0, x1*y1*z0], #110
                       [1., x0, y0, z1, x0*y0, x0*z1, y0*z1, x0*y0*z1], #001
                       [1., x1, y0, z1, x1*y0, x1*z1, y0*z1, x1*y0*z1], #101
                       [1., x0, y1, z1, x0*y1, x0*z1, y1*z1, x0*y1*z1], #011
                       [1., x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1], #111
                      ], dtype='float')

        invc      = inv(c)
        transinvc = np.transpose(invc)

        final = np.dot(transinvc, [1, x, y, z, x*y, x*z, y*z, x*y*z])

        interpFlux = 10**( (q000*final[0] + q100*final[1] + q010*final[2] + q110*final[3] + 
                            q001*final[4] + q101*final[5] + q011*final[6] + q111*final[7] ) )

        return interpFlux


    def GetModel(temp, logg, metal, modelset='marcs-apogee-dr15', wave=False):
        en = 0.00
        if instrument == 'nirspec':

            if modelset == 'btsettl08':
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            
            elif modelset == 'phoenixaces':
                filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
        
        elif instrument == 'apogee':
            
            if modelset == 'btsettl08':
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + '_en' + '{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
            
            elif modelset == 'phoenix-btsettl-cifist2011-2015':
                filename = 'PHOENIX_BTSETTL_CIFIST2011_2015_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

            elif modelset == 'phoenix-aces-agss-cond-2011':
                filename = 'PHOENIX_ACES_AGSS_COND_2011_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'
            
            elif modelset == 'marcs-apogee-dr15':
                filename = 'MARCS_APOGEE_DR15_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z{0:.2f}'.format(float(metal)) + '_en{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    

    if instrument == 'nirspec':

        if modelset == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'

        elif modelset == 'phoenixaces':
            Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'

    elif instrument == 'apogee':

        if modelset == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'

        elif modelset == 'phoenix-btsettl-cifist2011-2015':
            Gridfile = BASE + '/../libraries/PHOENIX_BTSETTL_CIFIST2011_2015/PHOENIX_BTSETTL_CIFIST2011_2015_gridparams_apogee.csv'

        elif modelset == 'phoenix-aces-agss-cond-2011':
            Gridfile = BASE + '/../libraries/PHOENIX_ACES_AGSS_COND_2011/PHOENIX_ACES_AGSS_COND_2011_gridparams_apogee.csv'

        elif modelset == 'marcs-apogee-dr15':
            Gridfile = BASE + '/../libraries/MARCS_APOGEE_DR15/MARCS_APOGEE_DR15_3D_gridparams_apogee.csv'

    T1 = Table.read(Gridfile)

    # Check if the model already exists (grid point)
    if (Teff, Logg, Metal) in zip(T1['Temp'], T1['Logg'], T1['Metal']): 
        index0 = np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg) & (T1['Metal'] == Metal) )
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
    ind000 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z0) ) # 000
    ind100 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z0) ) # 100
    ind010 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z0) ) # 010
    ind110 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z0) ) # 110
    ind001 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y0) & (T1['Metal'] == z1) ) # 001
    ind101 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y0) & (T1['Metal'] == z1) ) # 101
    ind011 = np.where( (T1['Temp'] == x0) & (T1['Logg'] == y1) & (T1['Metal'] == z1) ) # 011
    ind111 = np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1) & (T1['Metal'] == z1) ) # 111
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

