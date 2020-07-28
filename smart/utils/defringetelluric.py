import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import smart
import os, sys
import time
from astropy.table import Table
from astropy.io import fits
from scipy.signal import lfilter, firwin, freqz, fftconvolve
from scipy import fftpack

#########################################
## Parameters Set Up
#########################################
tell_data_name = 'jun01s0063'
order  = 33
#pixel_range_start, pixel_range_end = 0, -1#23

#method = 'wavelet'
method = 'hanningnotch'
#method = 'flatfilter'
PLOT   = True

# telluric data path
#path1     = os.getcwd()
#tell_path = path1 + '/nsdrp_out/fits/all/'
tell_path = '/Volumes/LaCie/nirspec/GJ628/20050601/nsdrp_out/fits/all'


# save to path
#save_to_path    = tell_data_name + '_defringe/O%s/'%order
save_to_path    =  '/Users/dinohsu/research/nirspec/test/defringe_telluric_fn/'+ tell_data_name + '_defringe/O%s_hanningnotch2/'%order

def defringetelluric():

if not os.path.exists(save_to_path):
    os.makedirs(save_to_path)

############################################
print(tell_data_name)

tell_sp = smart.Spectrum(name=tell_data_name, order=order, path=tell_path)

clickpoints = []
def onclick(event):
	print(event)
	global clickpoints
	clickpoints.append([event.xdata])
	print(clickpoints)
	plt.axvline(event.xdata, c='r', ls='--')
	plt.draw()
	if len(clickpoints) == 2:
		print('Closing Figure')
		plt.axvspan(clickpoints[0][0], clickpoints[1][0], color='0.5', alpha=0.5, zorder=-100)
		plt.draw()
		plt.pause(1)
		plt.close('all')


### Draw the figure with the power spectrum
cal1        = tell_sp.flux#[pixel_range_start:pixel_range_end]
xdim        = len(cal1)#[pixel_range_start:pixel_range_end])
nfil        = xdim//2 + 1
# Smooth the continuum
cal1smooth  = sp.ndimage.median_filter(cal1, size=30)
# Do the FFT
cal1fft     = fftpack.rfft(cal1-cal1smooth)
yp          = abs(cal1fft[0:nfil])**2
yp          = yp / np.max(yp)

fig, ax1 = plt.subplots(figsize=(12,6))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
freq        = np.arange(nfil)
yp[0:3]     = 0 # Fix for very low order noise
ax1.plot(freq, yp)
#ax1.axvline(f_high*2, c='r', ls='--')
#ax1.axvline(f_low*2, c='r', ls='--')
ax1.set_ylabel('Power Spectrum')
ax1.set_xlabel('1 / (1024 pix)')
ax1.set_title('Select the range you would like to filter out')
ax1.set_xlim(0, np.max(freq))
plt.show()
plt.close('all')
f_high   = np.max(clickpoints)/2
f_low    = np.min(clickpoints)/2


if method == 'wavelet':
	#### Wavelets
	from wavelets import WaveletAnalysis

	xdim      = len(tell_sp.flux)#[pixel_range_start:pixel_range_end])
	cal1      = tell_sp.flux#[pixel_range_start:pixel_range_end]

	# Smooth the continuum
	smoothed    = sp.ndimage.uniform_filter1d(cal1, 30)
	splinefit   = sp.interpolate.interp1d(np.arange(len(smoothed)), smoothed, kind='cubic')
	cal1smooth  = splinefit(np.arange(0, len(cal1))) #smoothed

	# use wavelets package: WaveletAnalysis
	enhance_row = cal1 - cal1smooth
	#print(enhance_row)

	dt     = 0.1
	wa     = WaveletAnalysis(enhance_row, dt=dt, axis=0)
	# wavelet power spectrum
	power  = wa.wavelet_power

	# scales
	scales = wa.scales
	# associated time vector
	t      = wa.time
	# reconstruction of the original data
	rx     = wa.reconstruction()

	defringe_data    = np.array(cal1.data, dtype=float)

	# reconstruct the fringe image
	#reconstruct_image      = np.zeros(defringe_data.shape)
	reconstruct_image      = np.real(rx)

	defringe_data -= reconstruct_image
	newSpectrum   = defringe_data

	if PLOT:

		fig = plt.figure(figsize=(12,6))
		ax1 = plt.subplot2grid((3, 1), (0, 0))
		ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
		
		#freq        = np.arange(nfil)
		ax1.plot(power**2)
		ax1.axvline(f_high*2, c='r', ls='--')
		ax1.axvline(f_low*2, c='r', ls='--')
		ax1.set_ylabel('Power Spectrum')
		ax1.set_xlabel('1 / (1024 pix)')
		#ax1.set_xlim(0, np.max(freq))
	
		ax2.plot(cal1[0:-23], label='original', alpha=0.5, lw=1, c='b')
		ax2.plot(newSpectrum[0:-23]+0.5*np.median(newSpectrum[0:-23]), label='defringed', alpha=0.8, lw=1, c='r')
		ax2.legend()
		ax2.set_ylabel('Flux')
		ax2.set_xlabel('Pixel')
		ax2.set_xlim(0, len(cal1[0:-23]))
		plt.tight_layout()
		plt.savefig(save_to_path+"defringed_spectrum.png", bbox_inches='tight')
		plt.show()



if method == 'hanningnotch':

	## REDSPEC version
	cal1        = tell_sp.flux#[pixel_range_start:pixel_range_end]
	xdim        = len(cal1)#[pixel_range_start:pixel_range_end])
	nfil        = xdim//2 + 1

	#print(xdim, nfil//2+1)
	freq        = np.arange(nfil//2+1) / (nfil / float(xdim))
	fil         = np.zeros(len(freq), dtype=np.float) 
	fil[np.where((freq < f_low) | (freq > f_high))] = 1.
	fil         = np.append(fil, np.flip(fil[1:],axis=0))
	fil         = np.real(np.fft.ifft(fil))
	fil         = np.roll(fil, nfil//2)
	fil         = fil*np.hanning(nfil)

	# Smooth the continuum
	#smoothed    = sp.ndimage.uniform_filter1d(cal1, 30)
	#splinefit   = sp.interpolate.interp1d(np.arange(len(smoothed)), smoothed, kind='cubic')
	#cal1smooth  = splinefit(np.arange(0, len(cal1))) #smoothed
	cal1smooth  = sp.ndimage.median_filter(cal1, size=30)

	"""
	plt.figure()
	plt.plot(abs(np.real(fftpack.fft(cal1orig-cal1smooth)))**2, c='k', lw=0.5)
	plt.plot(abs(np.real(fftpack.fft( sp.ndimage.convolve(cal1orig-cal1smooth, fil, mode='wrap') ) ))**2, c='r', lw=0.5)
	plt.ylim(0,25000)
	#plt.xlim(0,800)
	plt.show()
	#sys.exit()

	plt.figure(figsize=(10,6))
	plt.plot(cal1-cal1smooth, lw=0.5, c='k')
	plt.plot(sp.ndimage.convolve(cal1-cal1smooth, fil, mode='wrap'), lw=0.5, c='r')
	plt.plot(sp.ndimage.median_filter(cal1-cal1smooth, 10), lw=0.5, c='m')
	plt.show(block=False)
	#sys.exit()

	plt.figure()
	plt.plot( (cal1-cal1smooth)-sp.ndimage.convolve(cal1-cal1smooth, fil, mode='wrap'), lw=0.5, c='k')
	plt.show(block=True)
	#sys.exit()
	"""
	newSpectrum       = sp.ndimage.convolve(cal1-cal1smooth, fil, mode='wrap') + cal1smooth

	if PLOT:

		# Do the FFT
		cal1fft     = fftpack.rfft(cal1-cal1smooth)
		yp          = abs(cal1fft[0:nfil])**2
		yp          = yp / np.max(yp)
		yp[0:3]     = 0 # Fix for very low order noise

		fig = plt.figure(figsize=(12,6))
		ax1 = plt.subplot2grid((3, 1), (0, 0))
		ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
		
		freq        = np.arange(nfil)
		ax1.plot(freq, yp)
		ax1.axvline(f_high*2, c='r', ls='--')
		ax1.axvline(f_low*2, c='r', ls='--')
		ax1.set_ylabel('Power Spectrum')
		ax1.set_xlabel('1 / (1024 pix)')
		ax1.set_xlim(0, np.max(freq))
	
		ax2.plot(cal1[0:-23], label='original', alpha=0.5, lw=1, c='b')
		ax2.plot(newSpectrum[0:-23]+0.5*np.median(newSpectrum[0:-23]), label='defringed', alpha=0.8, lw=1, c='r')
		ax2.legend()
		ax2.set_ylabel('Flux')
		ax2.set_xlabel('Pixel')
		ax2.set_xlim(0, len(cal1[0:-23]))
		plt.tight_layout()
		plt.savefig(save_to_path+"defringed_spectrum.png", bbox_inches='tight')
		plt.show()



if method == 'flatfilter':

	cal1     = tell_sp.flux#[pixel_range_start:pixel_range_end]

	W        = fftpack.fftfreq(cal1.size, d=1./1024)
	fftval   = fftpack.rfft(cal1.astype(float))
	fftval[np.where((W > f_low) & (W < f_high))] = 0

	newSpectrum   = fftpack.irfft(fftval) 
	
	if PLOT: 

		xdim       = len(cal1)#[pixel_range_start:pixel_range_end])
		nfil       = xdim//2 + 1

		freq       = np.arange(nfil)

		# Smooth the continuum
		smoothed   = sp.ndimage.uniform_filter1d(cal1, 30)
		splinefit  = sp.interpolate.interp1d(np.arange(len(smoothed)), smoothed, kind='cubic')
		cal1smooth = splinefit(np.arange(0, len(cal1))) #smoothed

		# Do the FFT
		cal1fft    = fftpack.rfft(cal1-cal1smooth)
		yp         = abs(cal1fft[0:nfil])**2 # Power
		yp         = yp / np.max(yp)

		fig = plt.figure(figsize=(12,6))
		ax1 = plt.subplot2grid((3, 1), (0, 0))
		ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)

		ax1.plot(freq, yp)
		ax1.axvline(f_high, c='r', ls='--')
		ax1.axvline(f_low, c='r', ls='--')
		ax1.set_ylabel('Power Spectrum')
		ax1.set_xlabel('1 / (1024 pix)')
		ax1.set_xlim(0, np.max(freq))
	
		ax2.plot(cal1[0:-23], label='original', alpha=0.5, lw=1, c='b')
		ax2.plot(newSpectrum[0:-23]+0.5*np.median(newSpectrum[0:-23]), label='defringed', alpha=0.8, lw=1, c='r')
		ax2.legend()
		ax2.set_ylabel('Flux')
		ax2.set_xlabel('Pixel')
		ax2.set_xlim(0, len(cal1[0:-23]))
		plt.tight_layout()
		plt.savefig(save_to_path+"defringed_spectrum.png", bbox_inches='tight')
		#plt.show()


fullpath  = tell_path + '/' + tell_data_name + '_' + str(order) + '_all.fits'
save_name = save_to_path + '%s_defringe_%s_all.fits'%(tell_data_name, order)

hdulist = fits.open(fullpath)
hdulist.append(fits.PrimaryHDU())

hdulist[-1].data = tell_sp.flux
hdulist[1].data  = newSpectrum

hdulist[-1].header['COMMENT']  = 'Raw Extracted Spectrum'
hdulist[1].header['COMMENT']   = 'Defringed Spectrum'
try:
	hdulist.writeto(save_name, overwrite=True)
except FileNotFoundError:
	hdulist.writeto(save_name)






