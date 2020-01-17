#!/usr/bin/env python
#
# Nov. 17 2017
# @Dino Hsu
#
# This code is made to substract darks from flats 
# Given a folder, this function can create a list of
# flats and darks, perform dark subtraction from flats
# , and return a combined dark subtracted flat fits.
# Reference: NSDRP pipeline.

from astropy.io import fits
import numpy as np

###set up the parameters here
# Note this path is set before the running 3 digits
# ex: if the data filename is nov16s0001.fits, the path
#     to set up is 'nov16s'. The name of the dark 
#     subtracted flat would be nov16s_combine_flat.fits
path = ''
darkbegin = 1
darkend = 11
flatbegin = 12
flatend = 22

flatpath = path + format(flatbegin, '03d') + '.fits'
newflat = subtractDark(combineFlat(path, flatbegin, flatend)\
	, combineDark(path, darkbegin, darkend))

######

#save the dark-subtracted flat as a new fits file
data, header = fits.getdata(flatpath, header=True, ignore_missing_end=True)
data = newflat
if ('IMAGETYP' in header) is False:
	header['IMAGETYP'] = 'flatlamp'
if ('DISPERS' in header) is False:
	header['DISPERS'] = 'high'

flatend = flatend + 1
for i in range(flatbegin, flatend):
	j = i - flatbegin + 1
	keys = 'FLAT' + format(j,'01d')
	header[keys] = path + format(i, '03d')

header['CLEANED'] = 'Yes'

hdulist = fits.open(flatpath, ignore_missing_end=True)
hdulist[0].header = header
hdulist[0].data = newflat
newflatfilename = path + '_combine_flat.fits'
hdulist.writeto(newflatfilename, output_verify='ignore')
hdulist.close()


def combineDark(path, begin, end):
    """Median combine the dark frames
    """
    darkData = []
    end = end + 1
    for i in range(begin, end):
        fullpath = path + format(i , '03d') + '.fits'
        data, header = fits.getdata(fullpath, header=True, ignore_missing_end=True)
        darkData.append(data)
    combineDarks = np.median(darkData, axis=0)
    return combineDarks

def combineFlat(path, begin, end):
    """Median combine the flat frames
    """
    flatData = []
    end = end + 1
    for i in range(begin, end):
        fullpath = path + format(i , '03d') + '.fits'
        data, header = fits.getdata(fullpath, header=True, ignore_missing_end=True)
        flatData.append(data)
    combineFlats = np.median(flatData, axis=0)
    return combineFlats

def subtractDark(combineFlats, combineDarks):
    return np.subtract(combineFlats, combineDarks)

