import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import fits
from astropy.time import Time
import copy
import os
import shutil
import glob
from wavelets import WaveletAnalysis
import time
from astropy.visualization import ZScaleInterval, ImageNormalize

###################################################
# Defringe Flat method 
# adopted from ROJO & HARRINGTON 2006
###################################################

def defringeflat(flat_file, wbin=10, start_col=10, end_col=980, clip1=0,
	             diagnostic=True, save_to_path=None, filename=None):
	"""
	This function is to remove the fringe pattern using
	the method described in Rojo and Harrington (2006).

	Use a fifth order polynomial to remove the continuum.

	Parameters
	----------
	flat_file 		: 	fits
						original flat file

	Optional Parameters
	-------------------
	wbin 			:	int
						the bin width to calculate each 
						enhance row
						Default is 32

	start_col 		: 	int
						starting column number for the
						wavelet analysis
						Default is 10

	end_col 		: 	int
						ending column number for the
						wavelet analysis
						Default is 980

	diagnostic 		: 	boolean
						output the diagnostic plots
						Default is True

	Returns
	-------
	defringe file 	: 	fits
						defringed flat file

	"""
	# the path to save the diagnostic plots
	#save_to_path = 'defringeflat/allflat/'


	#print(flat_file)

	data = fits.open(flat_file, ignore_missing_end=True)

	data_length = len(data[0].data)

	date    = Time(data[0].header['DATE-OBS'], scale='utc')
	jd      = date.jd

	if jd >= 2458401.500000: # upgraded NIRSPEC
		# rotate to normal orientation
		data[0].data = np.rot90(data[0].data, k=3) # Fix for the upgraded NIRSPEC

	# Use the data to figure out the values to mask through the image (low counts/order edges)
	hist, bins = np.histogram(data[0].data.flatten(), bins=int(np.sqrt(len(data[0].data.flatten()))))
	bins       = bins[0:-1]
	index1     = np.where( (bins > np.percentile(data[0].data.flatten(),10)) & 
		                   (bins < np.percentile(data[0].data.flatten(),30)) )
	try:
		lowval = bins[index1][np.where(hist[index1] == np.min(hist[index1]))]
		#print(lowval, len(lowval))
		if len(lowval) >= 2: lowval = np.min(lowval)
	except:
		lowval = 0 #if no values for index1    

	
	flat = data

	# initial flat plot
	if diagnostic is True:
		
		# Save the images to a separate folder
		save_to_image_path = save_to_path + '/images/'
		if not os.path.exists(save_to_image_path):
			os.makedirs(save_to_image_path)

		fig = plt.figure(figsize=(8,8))
		fig.suptitle("original flat", fontsize=12)
		gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1]) 
		ax0 = plt.subplot(gs[0])
		# Create an ImageNormalize object
		norm = ImageNormalize(flat[0].data, interval=ZScaleInterval())
		ax0.imshow(flat[0].data, cmap='gray', norm=norm, origin='lower', aspect='auto')
		ax0.set_ylabel("Row number")
		ax1 = plt.subplot(gs[1], sharex=ax0)
		ax1.plot(flat[0].data[60,:],'k-',
			     alpha=0.5, label='60th row profile')
		ax1.set_ylabel("Amp (DN)")
		ax1.set_xlabel("Column number")
		plt.legend()
		plt.savefig(save_to_image_path + "defringeflat_{}_0_original_flat.png"\
			        .format(filename), bbox_inches='tight')
		plt.close()

	defringeflat_img = data
	defringe_data    = np.array(defringeflat_img[0].data, dtype=float)

	for k in np.arange(0, data_length-wbin, wbin):
		#print(k)
		#if k != 310: continue
		"""
		# Use the data to figure out the values to mask through the image (low counts/order edges)
		hist, bins = np.histogram(flat[0].data[k:k+wbin+1, 0:data_length-clip1].flatten(), 
			                      bins=int(np.sqrt(len(flat[0].data[k:k+wbin+1, 0:data_length-clip1].flatten()))))
		bins       = bins[0:-1]
		index1     = np.where( (bins > np.percentile(flat[0].data[k:k+wbin+1, 0:data_length-clip1].flatten(), 10)) & 
			                   (bins < np.percentile(flat[0].data[k:k+wbin+1, 0:data_length-clip1].flatten(), 30)) )
		lowval     = bins[index1][np.where(hist[index1] == np.min(hist[index1]))]
		
		#print(lowval, len(lowval))
		if len(lowval) >= 2: lowval = np.min(lowval)
		"""
		# Find the mask
		mask          = np.zeros(flat[0].data[k:k+wbin+1, 0:data_length-clip1].shape)
		baddata       = np.where(flat[0].data[k:k+wbin+1, 0:data_length-clip1] <= lowval)
		mask[baddata] = 1

		# extract the patch from the fits file
		#flat_patch = np.ma.array(flat[0].data[k:k+wbin,:], mask=mask)
		flat_patch = np.array(flat[0].data[k:k+wbin+1, 0:data_length-clip1])
		
		# median average the selected region in the order
		flat_patch_median = np.ma.median(flat_patch, axis=0)

		# continuum fit
		# smooth the continuum (Chris's method)
		smoothed  = sp.ndimage.uniform_filter1d(flat_patch_median, 30)
		splinefit = sp.interpolate.interp1d(np.arange(len(smoothed)), smoothed, kind='cubic')
		cont_fit  = splinefit(np.arange(0, data_length-clip1)) #smoothed

		# Now fit a polynomial
		#pcont     = np.ma.polyfit(np.arange(0, data_length-clip1),
		#	                      cont_fit, 10)
		#cont_fit2 = np.polyval(pcont, np.arange(0,data_length))
		
		#plt.plot(flat_patch_median, c='r')
		#plt.plot(smoothed, c='b')
		#plt.savefig(save_to_image_path + "TEST.png", bbox_inches='tight')
		#plt.close()
		#plt.show()
		#sys.exit()

		#pcont    = np.ma.polyfit(np.arange(start_col,end_col),
		#	                     flat_patch_median[start_col:end_col],10)
		#cont_fit = np.polyval(pcont, np.arange(0,data_length))

		# use wavelets package: WaveletAnalysis
		enhance_row = flat_patch_median - cont_fit

		dt = 0.1
		wa = WaveletAnalysis(enhance_row[start_col : end_col], dt=dt)
		# wavelet power spectrum
		power = wa.wavelet_power
		# scales
		cales = wa.scales
		# associated time vector
		t     = wa.time
		# reconstruction of the original data
		rx    = wa.reconstruction()

		# reconstruct the fringe image
		reconstruct_image = np.zeros(defringe_data[k:k+wbin+1, 0:data_length-clip1].shape)
		for i in range(wbin+1):
			for j in np.arange(start_col, end_col):
				reconstruct_image[i,j] = rx[j - start_col]

		defringe_data[k:k+wbin+1, 0:data_length-clip1] -= reconstruct_image[0:data_length-clip1]

		# Add in something for the edges/masked out data in the reconstructed image
		defringe_data[k:k+wbin+1, 0:data_length-clip1][baddata] = flat[0].data[k:k+wbin+1, 0:data_length-clip1][baddata]

		#print("{} row starting {} is done".format(filename,k))

		# diagnostic plots
		if diagnostic is True:
			print("Generating diagnostic plots")
			# middle cut plot
			fig = plt.figure(figsize=(10,6))
			fig.suptitle("middle cut at row {}".format(k+wbin//2), fontsize=12)
			ax1 = fig.add_subplot(2,1,1)

			norm = ImageNormalize(flat_patch, interval=ZScaleInterval())
			ax1.imshow(flat_patch, cmap='gray', norm=norm, origin='lower', aspect='auto')
			ax1.set_ylabel("Row number")
			ax2 = fig.add_subplot(2,1,2, sharex=ax1)
			ax2.plot(flat_patch[wbin//2,:],'k-',alpha=0.5)
			ax2.set_ylabel("Amp (DN)")
			ax2.set_xlabel("Column number")
			
			plt.tight_layout()
			plt.subplots_adjust(top=0.85, hspace=0.5)
			plt.savefig(save_to_image_path + \
				'defringeflat_{}_flat_start_row_{}_middle_profile.png'\
				.format(filename,k), bbox_inches='tight')
			plt.close()

        	# continuum fit plot
			fig = plt.figure(figsize=(10,6))
			fig.suptitle("continuum fit row {}-{}".format(k, k+wbin), fontsize=12)
			gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
			ax0 = plt.subplot(gs[0])
			ax0.plot(flat_patch_median, 'k-', alpha=0.5,
				     label='mean average patch')
			ax0.plot(cont_fit,'r-', alpha=0.5, label='continuum fit')
			#ax0.plot(cont_fit2,'m-', alpha=0.5, label='continuum fit poly')
			ax0.set_ylabel("Amp (DN)")
			plt.legend()
			ax1 = plt.subplot(gs[1])
			ax1.plot(flat_patch_median - cont_fit, 'k-',
				     alpha=0.5, label='residual')
			ax1.set_ylabel("Amp (DN)")
			ax1.set_xlabel("Column number")
			plt.legend()
			
			plt.tight_layout()
			plt.subplots_adjust(top=0.85, hspace=0.5)
			plt.savefig(save_to_image_path + \
			      	    "defringeflat_{}_start_row_{}_continuum_fit.png".\
				        format(filename,k), bbox_inches='tight')
			#plt.show()
			#sys.exit()
			plt.close()

			# enhance row vs. reconstructed wavelet plot
			try:
				fig = plt.figure(figsize=(10,6))
				fig.suptitle("reconstruct fringe comparison row {}-{}".\
					         format(k,k+wbin), fontsize=10)
				ax1 = fig.add_subplot(3,1,1)

				ax1.set_title('enhance_row start row')
				ax1.plot(enhance_row,'k-',alpha=0.5,
					     label="enhance_row start row {}".format(k))
				ax1.set_ylabel("Amp (DN)")
				#plt.legend()

				ax2 = fig.add_subplot(3,1,2, sharex=ax1)
				ax2.set_title('reconstructed fringe pattern')
				ax2.plot(rx,'k-',alpha=0.5,
					     label='reconstructed fringe pattern')
				ax2.set_ylabel("Amp (DN)")
				#plt.legend()

				ax3 = fig.add_subplot(3,1,3, sharex=ax1)
				ax3.set_title('residual')
				ax3.plot(enhance_row[start_col:end_col] - rx,
					     'k-',alpha=0.5, label='residual')
				ax3.set_ylabel("Amp (DN)")
				ax3.set_xlabel("Column number")
				#plt.legend()
				plt.tight_layout()
				plt.subplots_adjust(top=0.85, hspace=0.5)
				plt.savefig(save_to_image_path + \
					        "defringeflat_{}_start_row_{}_reconstruct_profile.png".\
					        format(filename,k), bbox_inches='tight')
				plt.close()
			except RuntimeError:
				print("CANNOT GENERATE THE PLOT defringeflat\
					_{}_start_row_{}_reconstruct_profile.png"\
					.format(filename,k))
				pass

			# reconstruct image comparison plot
			fig = plt.figure(figsize=(10,6))
			fig.suptitle("reconstructed image row {}-{}".\
				         format(k,k+wbin), fontsize=12)

			ax1 = fig.add_subplot(3,1,1)
			ax1.set_title('raw flat image')
			norm = ImageNormalize(flat_patch, interval=ZScaleInterval())
			ax1.imshow(flat_patch, cmap='gray', norm=norm, origin='lower',
				       label='raw flat image', aspect='auto')
			ax1.set_ylabel("Row number")
			#plt.legend()

			ax2 = fig.add_subplot(3,1,2, sharex=ax1)
			ax2.set_title('reconstructed fringe image')
			norm = ImageNormalize(reconstruct_image, interval=ZScaleInterval())
			ax2.imshow(reconstruct_image, cmap='gray', norm=norm, origin='lower',
				       label='reconstructed fringe image', aspect='auto')
			ax2.set_ylabel("Row number")
			#plt.legend()

			ax3 = fig.add_subplot(3,1,3, sharex=ax1)
			ax3.set_title('residual')
			norm = ImageNormalize(flat_patch-reconstruct_image, interval=ZScaleInterval())
			ax3.imshow(flat_patch-reconstruct_image, norm=norm, origin='lower',
				       cmap='gray', label='residual', aspect='auto')
			ax3.set_ylabel("Row number")
			ax3.set_xlabel("Column number")
			#plt.legend()
			plt.tight_layout()
			plt.subplots_adjust(top=0.85, hspace=0.5)
			plt.savefig(save_to_image_path + \
				        "defringeflat_{}_start_row_{}_reconstruct_image.png".\
				        format(filename,k), bbox_inches='tight')
			plt.close()

			# middle residual comparison plot
			fig = plt.figure(figsize=(10,6))
			fig.suptitle("middle row comparison row {}-{}".\
				         format(k,k+wbin), fontsize=12)

			ax1 = fig.add_subplot(3,1,1)
			ax1.plot(flat_patch[wbin//2,:],'k-',alpha=0.5,
				     label='original flat row {}'.format(k+wbin/2))
			ax1.set_ylabel("Amp (DN)")
			plt.legend()

			ax2 = fig.add_subplot(3,1,2, sharex=ax1)
			ax2.plot(flat_patch[wbin//2,:]-\
				     reconstruct_image[wbin//2,:],'k-',
				     alpha=0.5, label='defringed flat row {}'.format(k+wbin/2))
			ax2.set_ylabel("Amp (DN)")
			plt.legend()

			ax3 = fig.add_subplot(3,1,3, sharex=ax1)
			ax3.plot(reconstruct_image[wbin//2,:],'k-',alpha=0.5,
				     label='difference')
			ax3.set_ylabel("Amp (DN)")
			ax3.set_xlabel("Column number")
			plt.legend()
			
			plt.tight_layout()
			plt.subplots_adjust(top=0.85, hspace=0.5)
			plt.savefig(save_to_image_path + \
				        "defringeflat_{}_start_row_{}_defringe_middle_profile.png".\
				        format(filename,k), bbox_inches='tight')
			plt.close()

		#if k > 30: sys.exit() # for testing purposes

	# final diagnostic plot
	if diagnostic is True:
		fig = plt.figure(figsize=(8,8))
		fig.suptitle("defringed flat", fontsize=12)
		gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1]) 
		ax0 = plt.subplot(gs[0])
		norm = ImageNormalize(defringe_data, interval=ZScaleInterval())
		ax0.imshow(defringe_data, cmap='gray', norm=norm, origin='lower', aspect='auto')
		ax0.set_ylabel("Row number")
		ax1 = plt.subplot(gs[1],sharex=ax0)
		ax1.plot(defringe_data[60,:],'k-',
			     alpha=0.5, label='60th row profile')
		ax1.set_ylabel("Amp (DN)")
		ax1.set_xlabel("Column number")
		plt.legend()
		
		plt.tight_layout()
		plt.subplots_adjust(top=0.85, hspace=0.5)
		plt.savefig(save_to_image_path + "defringeflat_{}_0_defringe_flat.png"\
			.format(filename), bbox_inches='tight')
		plt.close()

	if jd >= 2458401.500000: # upgraded NIRSPEC
		# rotate back
		defringe_data = np.rot90(defringe_data, k=1)

	hdu = fits.PrimaryHDU(data=defringe_data)
	hdu.header = flat[0].header
	return hdu

def defringeflatAll(data_folder_path, wbin=10, start_col=10, 
	                end_col=980, diagnostic=True, movefiles=False):
	"""
	Perform the defringe flat function and save the 
	efringed flat files under the data folder and 
	move the raw flat files under anotehr folder 
	called "defringefalt_diagnostics" with optional 
	diagnostic plots.

	Parameters
	----------
	data_folder_path: 	str
						data folder for processing defringe flat

	Optional Parameters
	-------------------
	wbin 			:	int
						the bin width to calculate each 
						enhance row
						Default is 32

	start_col 		: 	int
						starting column number for the
						wavelet analysis
						Default is 10

	end_col 		: 	int
						ending column number for the
						wavelet analysis
						Default is 980

	diagnostic 		: 	boolean
						output the diagnostic plots
						The option may cause much more 
						computation time and have some issues
						in plotting.
						Default is True

	Returns
	-------
	defringe file 	: 	fits
						defringed flat file

	Examples
	--------
	>>> import nirspec_fmp as nsp
	>>> nsp.defringeflatAll(data_folder_path, diagnostic=False)

	"""
	originalpath = os.getcwd()

	save_to_path = data_folder_path + '/defringeflat/'
	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)

	# store the fits file names
	files = glob.glob1(data_folder_path,'*.fits')

	for filename in files:
		file_path = data_folder_path + filename

		data = fits.open(file_path, ignore_missing_end=True)

		if ('flat' in str(data[0].header['COMMENT']).lower()) is True or \
		('flatlamp' in str(data[0].header['IMAGETYP']).lower()) is True: # IMTYPE
			if ('flatlampoff' in str(data[0].header['IMAGETYP']).lower()) is True: continue
			if ('flat lamp off' in str(data[0].header['COMMENT']).lower()) is True: continue # dirty fix
			if ('dark for flat' in str(data[0].header['COMMENT']).lower()) is True: continue # dirty fix

			defringeflat_file = defringeflat(file_path, 
				wbin=wbin, start_col=start_col, 
				end_col=end_col ,diagnostic=diagnostic, 
				save_to_path=save_to_path,filename=filename)

			save_name = save_to_path + filename.split('.')[0] + \
			'_defringe.fits'
			if movefiles:
				save_name = data_folder_path + '/' + \
				filename.split('.')[0] + '_defringe.fits'
				shutil.move(data_folder_path + '/' + filename,
					save_to_path + filename)
			defringeflat_file.writeto(save_name, overwrite=True,
				output_verify='ignore')

	return None

#### Below is the test script for defringeflat function
## read in the median-combined flat
#flat =  fits.open('data/Burgasser/J0720-0846/2014jan19/reduced/flats/jan19s001_flat_0.fits')
#
#time1 = time.time()
#
#defringeflat(flat, diagnostic=False)
#
#time2 = time.time()
#print("total time: {} s".format(time2-time1))
