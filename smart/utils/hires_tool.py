import numpy as np
from astropy.io import fits

def compute_wavelength(header, order):
	"""
	Compute the wavelength from the MAKEE reduction output.

	References:
	https://www2.keck.hawaii.edu/inst/common/makeewww/Waves/index.html

	The 2D spectra (output by makee) have a wavelength scale for each echelle order specified by a 6th order polynomial of the form:
	wavelength in Angstroms = coef(0) + coef(1)*pixel + coef(2)*pixel*pixel + ... , where pixel is the column number.

	For each order, two FITS header cards are added. The two card names have the format: WV_0_## and WV_4_##
	where "##" is the echelle order number on a scale where the bluest order of a given exposure is number 01. 
	Each card contains four values:

	WV_0_## = ' coef(0) coef(1) coef(2) coef(3) '
	V_4_## = ' coef(4) coef(5) coef(6) coef(7) '

	For column number, where '0' is the first column in the internal image.

	Note that the HIRES MAKEE pipeline has corrected for heliocentric velocity (see the header at the end)

	Parameters
	----------
	header: fits header
			HIRES header info with MAKEE output

	order:  int
			order from the header

	Returns
	-------
	wave: 	numpy array
			wavelength array in Angstrom

	"""
	# read in the number of columns
	length = int(header['NAXIS1'])
	wave0  = list(filter(lambda a: a != '', header[f'WV_0_{str(order).zfill(2)}'].split(' ')))
	wave0.reverse()
	wave0  = [float(i) for i in wave0]

	wave1  = list(filter(lambda a: a != '', header[f'WV_4_{str(order).zfill(2)}'].split(' ')))
	wave1.reverse()
	wave1  = [float(i) for i in wave1]

	# combien the wavelength coefficients; reverse orders due to numpy polyval convention
	coeff  = wave1 + wave0

	wave   = np.polyval(coeff, np.arange(length))

	return wave

def create_fits_hires(filename, path, savepath=None):
	"""
	Combine the MAKEE data reduction product into a single fits file readable for SMART.
	"""

	# defult for savepath as the same for the path
	if savepath is None:
		savepath = path

	# read in the flux and error fits files
	flux   = fits.open(f'{path}{filename}_0_Flux.fits')
	error  = fits.open(f'{path}{filename}_0_Err.fits')
	header = flux[0].header
	orders = np.arange(header['NAXIS2']) + 1

	for order in orders:
		wave = compute_wavelength(header, order)

		# add wave, flux, error (noise)
		hdu  = fits.PrimaryHDU(data=wave, header=header)
		hdu1 = fits.PrimaryHDU(data=flux[0].data[order-1])
		hdu2 = fits.PrimaryHDU(data=error[0].data[order-1])

		# combine then into a master hdulist
		hdulist = fits.HDUList(hdu)
		hdulist.insert(1, hdu1)
		hdulist.insert(2, hdu2)

		# assign the header
		#hdulist[0].header = header

		savename = f'{filename}_{order}_all.fits'
		# save as the fits file per order
		hdulist.writeto(f'{savepath}{savename}', overwrite=True)

	return None

def get_hvsf(vhelio):
	"""
	MAKEE function that compute the heliocentric velocity scale factor (hvsf).

	The wavelength is multiplied by this scale factor with wave = wave * hvsf
	"""

	c = 299792.5 # not precise but this is what is defined in MAKEE hdr-util.f

	hvsf = np.sqrt( (1 + (vhelio/c) ) / ( 1-(vhelio/c) ) )

	return hvsf

def air2vac(wavelength):
	"""
	MAKEE function that converts the air to vacuum wavelength.

	Ported from astro_lib.f
	Assumes index of refraction of air with 15 deg c, 76 cm hg.
	Uses:  (n-1) x 10^7 = 2726.43 + 12.288/(w^2 x 10^-8) + 0.3555/(w^4 x 10^-16)
	"""

	x = wavelength

	n = ( 2726.43 + ( 12.288/( x**2 * 10**(-8)) ) + (0.3555/( x**4 * 10**(-16))) ) * 10**(-7) + 1.0

	y = (n - 1) * x

	air2vac_wave = x + y

	return air2vac_wave

def vac2air(wavelength):
	"""
	MAKEE function that converts the vacuum to air wavelength.

	Ported from astro_lib.f
	Assumes index of refraction of air with 15 deg c, 76 cm hg.
	Uses:  (n-1) x 10^7 = 2726.43 + 12.288/(w^2 x 10^-8) + 0.3555/(w^4 x 10^-16)
	"""

	x = wavelength

	n = ( 2726.43 + ( 12.288/( x**2 * 10**(-8)) ) + (0.3555/( x**4 * 10**(-16))) ) * 10**(-7) + 1.0

	y = (n - 1) * x

	vac2air_wave = x - y

	return vac2air_wave

def plot_spectra(sp, order_list):
	"""
	Plot multi-order HIRES spectra.
	
	"""
	pass