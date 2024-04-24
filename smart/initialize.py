# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import smart
from astropy.table import Table

ModelSets = {
    'phoenix-aces-agss-cond-2011' : 'PHOENIX_ACES_AGSS_COND_2011',
    'phoenix-btsettl-cifist2011' : 'PHOENIX_BTSETTL_CIFIST2011',
    'phoenix-btsettl-cifist2011-2015' : 'PHOENIX_BTSETTL_CIFIST2011_2015',
    'pheonix-newera-aces-cond-2023' : 'PHOENIX_NEWERA_ACES_COND_2023',
    'sonora' : 'SONORA_2018',
    'sonora-2023' : 'SONORA_2023',
    'phoenix-bt-dusty' : 'PHOENIX_BT_DUSTY',
    'drift-phoenix' : 'DRIFT_PHOENIX',
    'marcs-apogee-dr15' : 'MARCS_APOGEE_DR15',
    'marcs-pp' : 'MARCS_PP',
    'btsettl08' : 'BTSETTL08',
    'phoenix-btsettl08' : 'PHOENIX_BTSETTL08'
}

def getModelgrid(modelset = 'phoenix-aces-agss-cond-2011'):

    print('Retreiving gridfile for modelset: %s'%(modelset.upper()))
    # Get the gridfile for the requested modelset    
    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)
    Gridfile = BASE + '/libraries/%s/%s_gridparams.csv'%(smart.ModelSets[modelset.upper()], smart.ModelSets[modelset.upper()])
    # Read the grid file
    T1 = Table.read(Gridfile)

    return(T1)
