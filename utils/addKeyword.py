#!/usr/bin/env python
#
# Feb. 7 2017
# @Dino Hsu
#
# Given an input txt file of the data, this function 
# will add the required keywords for reduction via NSDRP

from astropy.io import fits
import os
import warnings
import numpy as np
import smart
warnings.filterwarnings("ignore")

def _addHeader(begin, end, date, imagetypes='object', debug=False):
	"""
	Add the keywords in the header such as dispers and imagetypes.
	@Dino Hsu

	Parameters
	----------
	begin: int number
	        the begin number of the data filename with the same imagetype
	end  : int number 
	       the end number of the data filename with the same imagetype
	date : str the date of the data filename
	       the date information in the filename
	imagetypes: str keyword 'IMAGETYP'
	       this can only be 'dark, flatlamp', or 'object'

	Returns
	-------
	Keywords 'IMAGETYP' and 'DISPERS' : keywords in fits header
	         the keywords required to put in the header 
	         before proccessing by NSDRP

	Notes
	-----
	1. DISPERS is always 'high', adding to the header all the time
	2. IMAGETYPES: NSDRP doesn't deal with the arclamp data.

	"""
	end = end + 1 #converge to the correct index of files

	for i in range(begin, end):
		files = date + 's0' + format(i, '03d') +'.fits'
		data, header = fits.getdata(files, header=True, ignore_missing_end=True)

		if ('IMAGETYP' in header) is False:
			header['IMAGETYP'] = imagetypes
		if ('DISPERS' in header) is False:
			header['DISPERS'] = 'high'

		#save the changes
		fits.writeto(files, data, header, overwrite=True, output_verify='ignore')
		
		#check if the keywords were added to the header correctly
		if debug is True:
			if ('IMAGETYP' in header) is True and ('DISPERS' in header) is True:
				print('The imagetype {0} and the high dispersion are added to the {1}'.format(imagetypes, files))

def add_nsdrp_keyword(file=input, debug=False):
	"""
	Add the required keywords for NSDRP reduction.
	@Dino Hsu

	Parameters
	----------
	file : txt format file 
	       Need to specify the data folder, date
	       darks, flats, arcs, and sources
	debug: True or Flase
	       Provide more details when adding the keywords

	Returns
	-------
	Keywords 'IMAGETYP' and 'DISPERS' : keywords in fits header
	         the keywords required to put in the header 
	         before proccessing by NSDRP

	Examples
	--------
	>>> from addKeyword import addKeyword
	>>> addKeyword(file='input_reduction.txt')

	"""
	with open(file="reduction_input.txt", mode="r") as f:
		table = f.readlines()
	# cd to the datafolder
	originalpath = os.getcwd()
	datafolder = table[0]
	yrdate = table[1]
	path = datafolder.split('DATAFOLDER\t',1)[1].split('\n',1)[0]\
	+yrdate.split()[1]
	date = table[1].split()[1][2:]
	os.chdir(path)

	# add the keywords
	for i in range(2,len(table)):
		keyword = table[i]
		begin = int(keyword.split()[1].split("-")[0])
		end = int(keyword.split()[1].split("-")[1])
		imagetype = str(keyword.split()[0])
		if imagetype == 'DARKS':
			_addHeader(begin=begin, end=end, date=date, imagetypes='dark', debug=debug)
		elif imagetype == 'FLATS':
			_addHeader(begin=begin, end=end, date=date, imagetypes='flatlamp', debug=debug)
		elif imagetype == 'ARC':
			_addHeader(begin=begin, end=end, date=date, imagetypes='arclamp', debug=debug)
		elif imagetype == 'SOURCE':
			_addHeader(begin=begin, end=end, date=date, imagetypes='object', debug=debug)
	os.chdir(originalpath)
	print('Keywords have been added, ready to be reduced by NSDRP.')

def add_wfits_keyword(filename, WFITS_filename, path, order):
	"""
	Add the WFITS parameters for initial guess of telluric wavelength calibrations.
	"""

	with fits.open(path+filename) as hdulist:
		with fits.open(path+WFITS_filename) as hdulist0:
			wfit0                         = hdulist0[0].header['WFIT0']
			wfit1                         = hdulist0[0].header['WFIT1'] 
			wfit2                         = hdulist0[0].header['WFIT2'] 
			wfit3                         = hdulist0[0].header['WFIT3'] 
			wfit4                         = hdulist0[0].header['WFIT4'] 
			wfit5                         = hdulist0[0].header['WFIT5'] 

			hdulist[0].header['COMMENT']  = 'Below are the keywords added by SMART package...'
			hdulist[0].header['WFIT0'] 	  = wfit0
			hdulist[0].header['WFIT1']    = wfit1
			hdulist[0].header['WFIT2']    = wfit2
			hdulist[0].header['WFIT3']    = wfit3
			hdulist[0].header['WFIT4']    = wfit4
			hdulist[0].header['WFIT5']    = wfit5
			hdulist[0].data               = smart.waveSolution(np.arange(len(hdulist[0].data)),
				                                           	wfit0, wfit1, wfit2, wfit3, 
				                                           	wfit4, wfit5, 0, 0, order=order)
			hdulist.writeto(path+filename, overwrite=True)







