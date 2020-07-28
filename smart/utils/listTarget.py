import os
import pandas as pd
from astropy.io import ascii, fits
import warnings
warnings.filterwarnings("ignore")

def makeTargetList(path, **kwargs):
	"""
	Return a csv file to store the KOA data information of sources.
	@Dino Hsu

	Parameters
	----------
	path      : str
	            The path of the data.

	Options
	-------
	outdir    : str
	            The path where the csv file will be stored.
	            The default is the same is the path of the data.

	save      : boolean value
			    True returns a csv file. The default is True.

	name 	  : str
				name of the csv file. The default is 'targetlist.csv'


	Returns
	-------
	targetlist: csv file format
			    including the header keywords
	            'OBJECT', 
	            'FILENAME', 
	            'SLITNAME', 
	            'DATE-OBS',
	            'PROGPI'

	data frame: pandas format data frame
	            including the same header keywords in the csv file.

	Examples
	--------
	>>> makeTargetList(path='path/to/PI_name')

	"""

	cwd    = os.getcwd()
	outdir = kwargs.get('outdir',path)
	save   = kwargs.get('save', True)
	name   = kwargs.get('name', 'targetlist')

	# crear empty lists to store keywords
	OBJECT_list   = []
	FILENAME_list = []
	SLITNAME_list = []
	DATEOBS_list  = []
	PROGPI_list   = []
	RA_list       = []
	DEC_list      = []

	os.chdir(path)
	listDataDir = os.listdir('.')

	for folder in listDataDir:
		path = folder + '/raw/spec/'
		null = ''
		try:
			filenames = os.listdir(path)
			for filename in filenames:
				fullpath = path + filename
				try:
					with fits.open(fullpath, ignore_missing_end=True) as f:
						if f[0].header['IMAGETYP'] == 'object':
							if f[0].header['OBJECT'] is not null and 'dark' not in f[0].header['OBJECT'] and\
							'arc' not in f[0].header['OBJECT'] and 'flat' not in f[0].header['OBJECT'] and\
							'test' not in f[0].header['OBJECT'] and 'Dark' not in f[0].header['OBJECT'] and\
							'Arc' not in f[0].header['OBJECT'] and 'Flat' not in f[0].header['OBJECT'] and\
							'Test' not in f[0].header['OBJECT'] and 'null' not in f[0].header['RA'] and\
							'null' not in f[0].header['DEC']:
								OBJECT_list.append(f[0].header['OBJECT'])
								FILENAME_list.append(f[0].header['FILENAME'])
								SLITNAME_list.append(f[0].header['SLITNAME'])
								DATEOBS_list.append(f[0].header['DATE-OBS'])
								PROGPI_list.append(f[0].header['PROGPI'])
								RA_list.append(f[0].header['RA'])
								DEC_list.append(f[0].header['DEC'])
							else:
								pass
						else:
							pass
				except IOError:
					pass
				except IsADirectoryError:
					pass
				except NotADirectoryError:
					pass
				except FileNotFoundError:
					pass
		except NotADirectoryError:
			pass
		except FileNotFoundError:
			pass

	df = pd.DataFrame({"OBJECT" : OBJECT_list, "FILENAME" : FILENAME_list, \
		"SLITNAME": SLITNAME_list, "DATE-OBS": DATEOBS_list, "PROGPI":PROGPI_list, \
		"RA":RA_list, "DEC":DEC_list})
	df = df.reindex_axis(['OBJECT','PROGPI','DATE-OBS','FILENAME','SLITNAME','RA','DEC'], axis=1)
	
	# save as a csv file
	if save == True:
		save_to_path = outdir + name + '.csv'
		df.to_csv(save_to_path, index=False)
		print("The target list is saved to {} .".format(save_to_path))

	os.chdir(cwd)

	return df