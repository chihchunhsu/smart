import smart
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
from . import tellurics

#--------------------------------------------
# 20200224 chsu: routine to generate the mask 
#--------------------------------------------

def telluric_mask(data, sigma=2.5, lsf=4.8, pwv=None, pixel_start=10, pixel_end=-30, outlier_rejection=2.5, diagnostic=True, save_to_path='./'):
	"""
	Routine to generate a mask for tellurics as the MCMC initialization.

	Parameters
	----------
	sigma 				: 	float; default 2.5
							The sigma-clipping method to reject the outliers.

	lsf 				:	float; default 4.8
							The value of line spread function. 
							The default is the normal value of LSF for Keck/NIRSPEC.

	pwv 				: 	float; default None
							precitable water vapor.
							The default is to run the chi2 grids of pwv to obtain the best pwv.

	pixel_start			: 	int; default 10
							The starting pixel to compute the mask.

	pixel_end 			: 	int; default -30
							The ending pixel to compute the mask.

	outlier_rejection 	: 	float; default 2.5
							The value for number * sigma to reject outliers

	diagnostic 			: 	bool; default True
							Generate the diagnostic plots for pwv-ch2 and model-data before/after masks

	save_to_path 		: 	str; default the current working directory


	Returns
	-------
	mask 				: numpy array

	pwv 				: float 

	airmass 			: float


	Example
	-------
	>>> from smart.forward_model import mask
	>>> tell_mask, pwv, airmass = mask.telluric_mask(data, diagnostic=False)

	"""

	data.flux  = data.flux[pixel_start:pixel_end]
	data.wave  = data.wave[pixel_start:pixel_end]
	data.noise = data.noise[pixel_start:pixel_end]

	data0 = copy.deepcopy(data)

	# take the closest airmass from the header
	airmass = float(round(data.header['AIRMASS']*2)/2)
	if airmass > 3.0: airmass = 3.0

	# simple chi2 comparison with different pwv
	if pwv is None:
		pwvs = [0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0]
		pwv_chi2 = []
		
		for pwv in pwvs:
			data_tmp       = copy.deepcopy(data)
	
			#data_tmp       = smart.continuumTelluric(data=data_tmp, model=model_tmp)
			model_tmp      = tellurics.makeTelluricModel(lsf=lsf, airmass=airmass, pwv=pwv, flux_offset=0, wave_offset=0, data=data_tmp)
	
			model_tmp.flux = np.array(smart.integralResample(xh=model_tmp.wave, yh=model_tmp.flux, xl=data_tmp.wave))
			model_tmp.wave = data_tmp.wave

			#plt.plot(data_tmp.wave, data_tmp.flux, 'k-')
			#plt.plot(model_tmp.wave, model_tmp.flux, 'r-')
			#plt.show()
			#plt.close()
	
			pwv_chi2.append(smart.chisquare(data_tmp, model_tmp))
		# find the pwv with minimum chisquare
		pwv_chi2_array = np.array(pwv_chi2)
		
		if diagnostic:
			plt.plot(pwvs, pwv_chi2)
			plt.xlabel('pwv (mm)', fontsize=15)
			plt.ylabel('$\chi^2$', fontsize=15)
			plt.tight_layout()
			plt.savefig(save_to_path+'pwv_chi2.png'.format(order))
			#plt.show()
			plt.close()

	pwv_min_index = np.where(pwv_chi2_array == np.min(pwv_chi2_array))[0][0]
	pwv           = pwvs[pwv_min_index]

	data_tmp  = copy.deepcopy(data)

	model      = tellurics.makeTelluricModel(lsf=lsf, airmass=airmass, pwv=pwv, flux_offset=0, wave_offset=0, data=data_tmp)

	model_0 = copy.deepcopy(model)

	# generate the mask based on sigma clipping
	pixel = np.delete(np.arange(len(data.oriWave)), data.mask)[pixel_start: pixel_end]
	#pixel = np.delete(np.arange(len(data_tmp.oriWave)),data_tmp.mask)
	mask  = pixel[np.where(np.abs(data_tmp.flux-model.flux) > outlier_rejection*np.std(data_tmp.flux-model.flux))]

	#plt.plot(data_tmp.wave, data_tmp.flux, 'k-')
	#plt.plot(model.wave, model.flux, 'r-')
	#plt.show()
	#plt.close()

	data_tmp.mask_custom(mask)
	data_tmp.flux  = data_tmp.flux[pixel_start:pixel_end]
	data_tmp.wave  = data_tmp.wave[pixel_start:pixel_end]
	data_tmp.noise = data_tmp.noise[pixel_start:pixel_end]

	#plt.plot(data_tmp.wave, data_tmp.flux, 'k-')
	#plt.plot(model.wave, model.flux, 'r-')
	#plt.show()
	#plt.close()

	# use curve_fit
	def tell_model_fit(wave, airmass, pwv, flux_offset, wave_offset):
		model      = tellurics.makeTelluricModel(lsf=lsf, airmass=airmass, pwv=pwv, flux_offset=flux_offset, wave_offset=wave_offset, data=data_tmp)
		return model.flux

	print('initial airmass and pwv', airmass, pwv)
	flux_med = np.median(data_tmp.flux)
	p0 = [airmass, pwv, 0, 0]
	bounds = ([airmass-0.5, pwv-0.5, -flux_med*0.05, -0.05], [airmass+0.5, pwv+0.5, flux_med*0.05, 0.05])

	popt, pcov = curve_fit(tell_model_fit, data_tmp.wave, data_tmp.flux, p0=p0, bounds=bounds)

	airmass, pwv, flux_offset, wave_offset = popt[0], popt[1], popt[2], popt[3]
	print('best-fit airmass, pwv, flux_offset, wave_offset', airmass, pwv, flux_offset, wave_offset)
	model      = tellurics.makeTelluricModel(lsf=lsf, airmass=airmass, pwv=pwv, flux_offset=0, wave_offset=0, data=data_tmp)
	print('old telluric mask', mask)
	pixel = np.delete(np.arange(len(data_tmp.oriWave)), mask)[pixel_start: pixel_end]
	#print('len pixel, data, model', len(pixel), len(data_tmp.wave), len(model.wave))
	mask  = pixel[np.where(np.abs(data_tmp.flux-model.flux) > outlier_rejection*np.std(data_tmp.flux-model.flux))]
	# combine the masks
	mask = np.union1d(mask,np.array(data_tmp.mask))
	print('new telluric mask', mask)

	data.mask_custom(mask)
	data.flux  = data.flux[pixel_start:pixel_end]
	data.wave  = data.wave[pixel_start:pixel_end]
	data.noise = data.noise[pixel_start:pixel_end]
	print(data.mask)

	#plt.plot(data.wave, data.flux, 'k-')
	#plt.plot(model.wave, model.flux, 'r-')
	#plt.plot(model_0.wave, model_0.flux, 'b-', alpha=0.5)
	#plt.show()
	#plt.close()

	if diagnostic:
		data.flux  = data.flux[pixel_start:pixel_end]
		data.wave  = data.wave[pixel_start:pixel_end]
		data.noise = data.noise[pixel_start:pixel_end]
		
		model      = tellurics.makeTelluricModel(lsf=lsf, airmass=airmass, pwv=pwv, flux_offset=0, wave_offset=0, data=data)
		
		plt.plot(data0.wave, data0.flux, 'k-', label='original data', alpha=0.5)
		plt.plot(data.wave, data.flux, 'k-', label='masked data')
		plt.plot(model.wave, model.flux, 'r-', alpha=0.7)
		plt.plot(data.wave, data.flux-model.flux, 'r-')
		plt.xlabel('$\lambda (\AA)$')
		plt.ylabel('$F_{\lambda}$')
		plt.savefig(save_to_path+'telluric_data_model_mask.png')
		#plt.show()
		plt.close()

	return mask, pwv, airmass

