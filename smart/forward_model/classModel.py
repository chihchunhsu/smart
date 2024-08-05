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

#def _constructModelName(teff, logg, metal, en, order, path=None):
#    """
#    Return the full name of the BT-Settl model.
#    """
#    if path is None:
#        path  = '/Users/dinohsu/projects/Models/models/btsettl08/' + \
#        'NIRSPEC-O' + str(order) + '-RAW/'
#    else:
#        path  = path + '/NIRSPEC-O' + str(order) + '-RAW/'
#    full_name = path + 'btsettl08_t'+ str(teff) + '_g' + \
#    '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(metal)) + \
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
    teff : float 
          The effective temperature in Kelvins.
    logg : float
          The log(gravity), given in two decimal digits. 
          Ex: logg=4.50
    metal  : float
           The metalicity, given in two decimal digits. 
           Ex. metal=0.00
    en   : float
           alpha enhancement. given in two decimal digits. 
           Ex. en=0.00

    modelset: str
            available models are 
            NIRSPEC: 'btsettl08', 'SONORA_2018'
            APOGEE: 'btsettl08', 'marcs-apogee-dr15', 'phoenix-aces-agss-cond-2011', 'phoenix-btsettl-cifist2011-2015'

    order: int
           This is only for the Keck/NIRSPEC. The order of the model, given from 29 to 80

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
           The flux retrieved from the model. Our default unit is erg/s/cm^2/Angstrom.
    wave : astropy.table.column.Column
           The wavelength retrieved from the model. Our default unit is Angstrom.

    Examples
    --------
    >>> import smart
    >>> model = smart.Model(teff=2300, logg=5.5, order=33, path='/path/to/models')
    >>> model.plot()
    """
    def __init__(self, **kwargs):
        self.path  = kwargs.get('path')
        self.order = kwargs.get('order')
        self.instrument = kwargs.get('instrument','nirspec')

        if (self.order != None) and (self.instrument in ['nirspec', 'hires', 'igrins']):
            self.teff     = kwargs.get('teff', 2500)
            self.logg     = kwargs.get('logg', 5.00)
            self.metal    = kwargs.get('metal', 0.00)
            self.en       = kwargs.get('en', 0.00)
            self.modelset = kwargs.get('modelset', 'btsettl08')

            wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg, self.metal, self.en,
                                                                          modelset=self.modelset, order=self.order, instrument=self.instrument)
            #elif self.metal != 0.0:
            #    wave, flux = smart.forward_model.InterpolateModel.InterpModel(Teff=self.teff, Logg=self.logg, Metal=self.metal,
            #    modelset=self.modelset, order=self.order, instrument=self.instrument)
            
            if self.modelset == 'btsettl08':
                self.wave = wave * 10000 #convert to Angstrom
                self.flux = flux / 10000 #convert from erg/s/cm^2/micron to erg/s/cm^2/Angstrom
            else:
                self.wave = wave
                self.flux = flux

        elif self.instrument == 'apogee':
            self.teff     = kwargs.get('teff', 2500)
            self.logg     = kwargs.get('logg', 5.00)
            self.metal    = kwargs.get('metal', 0.00)
            self.en       = kwargs.get('en', 0.00)
            self.modelset = kwargs.get('modelset', 'btsettl08')

            #wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg,
            #    modelset=self.modelset, order=self.order, instrument=self.instrument)

            wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg, self.metal, self.en,
                                                                          modelset=self.modelset, order=self.order, instrument=self.instrument)

            if self.modelset == 'btsettl08':
                self.wave = wave * 10000 #convert to Angstrom
                self.flux = flux / 10000 #convert from erg/s/cm^2/micron to erg/s/cm^2/Angstrom 

            else:
                self.wave = wave # Angstrom
                self.flux = flux # erg/s/cm^2/Angstrom

        elif self.order == None and self.instrument in ['lowres1', 'lowres5', 'lowres10', 'lowres100']:

            self.teff     = kwargs.get('teff', 2500)
            self.logg     = kwargs.get('logg', 5.00)
            self.metal    = kwargs.get('metal', 0.00)
            self.en       = kwargs.get('en', 0.00)
            self.modelset = kwargs.get('modelset', 'btsettl08')

            wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg, self.metal, self.en,
                                                                          modelset=self.modelset, order='', instrument=self.instrument)
            self.wave = wave # Angstrom
            self.flux = flux # erg/s/cm^2/Angstrom

        else:
            try:
                self.teff     = kwargs.get('teff', 2500)
                self.logg     = kwargs.get('logg', 5.00)
                self.metal    = kwargs.get('metal', 0.00)
                self.en       = kwargs.get('en', 0.00)
                self.kzz      = kwargs.get('kzz', 0.00)
                self.co       = kwargs.get('co', 0.00)
                self.modelset = kwargs.get('modelset', 'btsettl08')

                wave, flux = smart.forward_model.InterpolateModel.InterpModel(self.teff, self.logg, self.metal, self.en, self.kzz, self.co,
                                                                              modelset=self.modelset, order=self.order, instrument=self.instrument)
                self.wave = wave # Angstrom
                self.flux = flux # erg/s/cm^2/Angstrom
            except:
                self.wave   = kwargs.get('wave', [])
                self.flux   = kwargs.get('flux', [])
        

    def normalize(self, filter_size=500, **kwargs):
        """
        Normalize the continuum by fitting a linear slope to local max over a range of model fluxes, parameterized by filter_size.
        """
        max_list = []
        for i in range(int(len(self.wave)/filter_size)):
            max_list.append(max(self.flux[i*filter_size:(i+1)*filter_size]))

        x, y = self.wave[[i*filter_size+int(filter_size/2) for i in range(int(len(self.wave)/500))]], max_list
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        self.flux = self.flux/p(self.wave)


    def plot(self, **kwargs):
        """
        Plot the model spectrum.
        """
        if self.order != None:
            name = str(_constructModelName(self.teff, self.logg, 
                self.metal, self.en, self.order, self.path))
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


