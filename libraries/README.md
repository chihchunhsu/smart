NIRSPEC wavelengths are in microns.
NIRSPEC fluxes are in erg / (s * m^2 * angstrom)

APOGEE wavelengths are in angstrom.
APOGEE fluxes are in erg / (s * cm^2 * angstrom)

The library includes 

* BT-Settl08 Models (Allard et al. 2010) order 33, which can be read in nirspec_fmp.Model()

* high resolution (300k) telluric models (Moefler et al. 2014), which can be read in nirspec_fmp.getTelluric()
