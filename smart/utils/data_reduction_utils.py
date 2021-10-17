import numpy as np
from astropy.io import fits
import smart

def rotate_data(data):
	"""
	Routine for rotating the upgraded NIRSPEC data to the normal orientation.
	"""
	new_data = np.rot90(np.rot90(np.rot90(data)))

	return new_data


def update_header_keyword(header):
	"""
	Routine for updating the header keyword "IMTYPE" to "IMAGETYP".
	"""
	if 'IMTYPE' in header and 'IMAGETYP' not in header:
		header['IMAGETYP'] = header['IMTYPE']

	if 'DISPERS' not in header:
		header['DISPERS'] = 'high'

	return header


def create_master_flat(folder_path):
	
	pass


def update_wfits_param(filename_A:str, filename_B:str, order:int, data_path:str, data_path2:str=None, verbose:bool=True) -> None:
	"""
	Update NIRSPEC spectra file A WFITS parameters using file B info. 

	WFITS parameters are necessary for telluric wavelength calibration. 
	Typically file A is telluric standard with a short exposure time and the WFITS parameters cannot be determined from NSDRP
	and this can be fixed with using another file B is the science frame with longer exposure.

	Parameters
	----------
	filename_A: str; the file name to be updated with WFITS parameters
	filename_B: str; the file name to update the file A using its WFITS parameters
	order:		int; NIRSPEC order
	data_path:	str; path of the file A, if data_path2 is None, the file B is assumed to have the same file path as file A
	data_path2:	str; path of the file B
	
	"""

	data_pathA = data_path + '/' + filename_A + '_' + str(order) + '_all.fits'
	
	if data_path2 is not None:
		data_pathB = data_path2 + '/' + filename_B + '_' + str(order) + '_all.fits'
	else:
		data_pathB = data_path + '/' + filename_B + '_' + str(order) + '_all.fits'

	with fits.open(data_pathA) as hdulist:
		with fits.open(data_pathB) as hdulist0:
			length1                       = len(hdulist[0].data) # set up the length for pre-/post-upgrade NIRSPEC

			wfit0                         = hdulist0[0].header['WFIT0']
			wfit1                         = hdulist0[0].header['WFIT1'] 
			wfit2                         = hdulist0[0].header['WFIT2'] 
			wfit3                         = hdulist0[0].header['WFIT3'] 
			wfit4                         = hdulist0[0].header['WFIT4'] 
			wfit5                         = hdulist0[0].header['WFIT5'] 

			hdulist[0].header['WFIT0'] 	  = wfit0
			hdulist[0].header['WFIT1']    = wfit1
			hdulist[0].header['WFIT2']    = wfit2
			hdulist[0].header['WFIT3']    = wfit3
			hdulist[0].header['WFIT4']    = wfit4
			hdulist[0].header['WFIT5']    = wfit5
			hdulist[0].data               = smart.waveSolution(np.arange(length1),
				wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, 0, 0, order=order)
			hdulist.writeto(data_pathA, overwrite=True)
			if verbose:
				print(f'The WFITS parameters of file {filename_A} has been updated using {filename_B} in order {order}.')

