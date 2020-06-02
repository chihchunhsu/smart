#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import smart
#import splat
#import splat.model as spmd

#def _constructModelName(teff, logg, feh, en, order, path=None):
#    """
#    Return the full name of the BT-Settl model.
#    """
#    if path is None:
#        path  = '/Users/dinohsu/projects/Models/models/btsettl08/' + \
#        'NIRSPEC-O' + str(order) + '-RAW/'
#    else:
#        path  = path + '/NIRSPEC-O' + str(order) + '-RAW/'
#    full_name = path + 'btsettl08_t'+ str(teff) + '_g' + \
#    '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + \
#    '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
#    
#    return full_name

class Model():
    """
    The Model class reads in the BT-SETTL or PHOENIXACES models. 
    The unit of wavelength is in Angstrom and the unit of the model flux is in erg/s/cm^2/Angstrom.
    (The models in the libraries have the unit of micron, which differed by 10^4 in flux)

    Parameters
    ----------
    1. Read in a BT-Settl model or PHOENIXACES model.
    teff : int 
          The effective temperature, given from 500 to 3,500 K.
    logg : float
          The log(gravity), given in two decimal digits. 
          Ex: logg=4.50
    feh  : float
           The metalicity, given in two decimal digits. 
           Ex. feh=0.00
    en   : float
           alpha enhancement. given in two decimal digits. 
           Ex. en=0.00
    order: int
           The order of the model, given from 29 to 80
    path : str
           The path to the model

    2. Creat a model instance with given wavelengths and fluxes
    flux : astropy.table.column.Column
           The input flux.
    wave : astropy.table.column.Column
           The input wavelength.

    Returns
    -------
    flux : astropy.table.column.Column
           The flux retrieved from the model.
    wave : astropy.table.column.Column
           The wavelength retrieved from the model.

    Examples
    --------
    >>> import nirspec_pip as smart
    >>> model = smart.Model(teff=2300, logg=5.5, order=33, path='/path/to/models')
    >>> model.plot()
    """
    def __init__(self, **kwargs):
        self.path  = kwargs.get('path')
        self.order = kwargs.get('order')
        self.instrument = kwargs.get('instrument','nirspec')

        if self.order != None and self.instrument == 'nirspec':
            self.teff     = kwargs.get('teff', 2500)
            self.logg     = kwargs.get('logg', 5.00)
            self.feh      = kwargs.get('feh', 0.00)
            self.en       = kwargs.get('en', 0.00)
            self.modelset = kwargs.get('modelset', 'btsettl08')

            #print('Return a BT-Settl model of the order {0}, with Teff {1} logg {2}, z {3}, Alpha enhancement {4}.'\
            #    .format(self.order, self.teff, self.logg, self.feh, self.en))
        
            #full_name = _constructModelName(self.teff, self.logg, self.feh, self.en, self.order, self.path)
            #model = ascii.read(full_name, format='no_header', fast_reader=False)
            #self.wave  = model[0][:]*10000 #convert to Angstrom
            #self.flux  = model[1][:]
            
            ## load the splat.interpolation BTSETTL model
            #instrument = "NIRSPEC-O{}-RAW".format(self.order)
            #sp = spmd.getModel(instrument=str(instrument),teff=self.teff,logg=self.logg,z=self.feh)
            #self.wave = sp.wave.value*10000 #convert to Angstrom
            #self.flux = sp.flux.value

            wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg,
                modelset=self.modelset, order=self.order, instrument=self.instrument)
            if self.modelset == 'btsettl08':
                self.wave = wave * 10000 #convert to Angstrom
                self.flux = flux / 10000 #convert from erg/s/cm^2/micron to erg/s/cm^2/Angstrom
            else:
                self.wave = wave
                self.flux = flux

        elif self.instrument == 'apogee':
            self.teff     = kwargs.get('teff', 2500)
            self.logg     = kwargs.get('logg', 5.00)
            self.feh      = kwargs.get('feh', 0.00)
            self.en       = kwargs.get('en', 0.00)
            self.modelset = kwargs.get('modelset', 'btsettl08')

            #wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg,
            #    modelset=self.modelset, order=self.order, instrument=self.instrument)

            wave, flux = smart.forward_model.InterpolateModel.InterpModel_3D(Teff=self.teff, Logg=self.logg, Metal=self.feh,
                modelset=self.modelset, order=self.order, instrument=self.instrument)

            self.wave = wave # Angstrom
            self.flux = flux # erg/s/cm^2/Angstrom

        else:
            self.wave   = kwargs.get('wave', [])
            self.flux   = kwargs.get('flux', [])
        

    def plot(self, **kwargs):
        """
        Plot the model spectrum.
        """
        if self.order != None:
            name = str(_constructModelName(self.teff, self.logg, 
                self.feh, self.en, self.order, self.path))
            output = kwargs.get('output', str(name) + '.pdf')
            ylim = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
            title  = kwargs.get('title')
            save   = kwargs.get('save', False)
        
            plt.figure(figsize=(16,6))
            plt.plot(self.wave, self.flux, color='k', 
                alpha=.8, linewidth=1, label=name)
            plt.legend(loc='upper right', fontsize=12)
            plt.ylim(ylim)    
    
            minor_locator = AutoMinorLocator(5)
            #ax.xaxis.set_minor_locator(minor_locator)
            # plt.grid(which='minor') 
    
            plt.xlabel(r'$\lambda$ [$\mathring{A}$]', fontsize=18)
            plt.ylabel(r'$Flux$', fontsize=18)
            #plt.ylabel(r'$F_{\lambda}$ [$erg/s \cdot cm^{2}$]', fontsize=18)
            if title != None:
                plt.title(title, fontsize=20)
            plt.tight_layout()

            if save == True:
                plt.savefig(output)
            plt.show()
            plt.close()

        else:
            output = kwargs.get('output'+ '.pdf')
            ylim   = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
            title  = kwargs.get('title')
            save   = kwargs.get('save', False)
        
            plt.figure(figsize=(16,6))
            plt.plot(self.wave, self.flux, color='k', alpha=.8, linewidth=1)
            plt.legend(loc='upper right', fontsize=12)
            plt.ylim(ylim)
    
            minor_locator = AutoMinorLocator(5)
            #ax.xaxis.set_minor_locator(minor_locator)
            # plt.grid(which='minor') 
    
            plt.xlabel(r'$\lambda$ [$\mathring{A}$]', fontsize=18)
            plt.ylabel(r'$Flux$', fontsize=18)
            #plt.ylabel(r'$F_{\lambda}$ [$erg/s \cdot cm^{2}$]', fontsize=18)
            if title != None:
                plt.title(title, fontsize=20)
            plt.tight_layout()

            if save == True:
                plt.savefig(output)
            plt.show()
            plt.close()


