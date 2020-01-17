import numpy as np
from astropy.io import fits

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