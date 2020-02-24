import smart
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
from . import tellurics

#--------------------------------------------
# 20200223 chsu: routine to generate the mask 
#--------------------------------------------

def telluric_mask(data, sigma=2.5, lsf=4.8, pwv=None, pixel_start=10, pixel_end=-50, outlier_rejection=2.5, diagnostic=True):
	"""
	"""

	data.flux  = data.flux[pixel_start:pixel_end]
	data.wave  = data.wave[pixel_start:pixel_end]
	data.noise = data.noise[pixel_start:pixel_end]

	data0 = copy.deepcopy(data)

	# take the closest airmass from the header
	airmass = float(round(data.header['AIRMASS']*2)/2)
	if airmass > 3.0: airmass = 3.0
	airmass  = str(airmass)

	# simple chi2 comparison with different pwv
	if pwv is None:
		pwvs = ['0.5', '1.0', '1.5', '2.5', '3.5', '5.0', '7.5', '10.0', '20.0']
		pwv_chi2 = []

		#wavelow  = data.wave[0] - 5
		#wavehigh = data.wave[-1] + 5
		
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
		
		plt.plot(pwvs, pwv_chi2)
		plt.xlabel('pwv (mm)', fontsize=15)
		plt.ylabel('$\chi^2$', fontsize=15)
		plt.tight_layout()
		#plt.savefig('pwv_chi2.png'.format(order))
		plt.show()
		plt.close()

	pwv_min_index = np.where(pwv_chi2_array == np.min(pwv_chi2_array))[0][0]
	pwv           = pwvs[pwv_min_index]

	data_tmp  = copy.deepcopy(data)

	model      = tellurics.makeTelluricModel(lsf=lsf, airmass=airmass, pwv=pwv, flux_offset=0, wave_offset=0, data=data_tmp)

	#data_tmp       = smart.continuumTelluric(data=data_tmp, model=model_tmp)

	# generate the mask based on sigma clipping
	pixel = np.delete(np.arange(len(data.oriWave)), data.mask)[pixel_start: pixel_end]
	#pixel = np.delete(np.arange(len(data_tmp.oriWave)),data_tmp.mask)
	mask  = pixel[np.where(np.abs(data_tmp.flux-model.flux) > outlier_rejection*np.std(data_tmp.flux-model.flux))]

	plt.plot(data_tmp.wave, data_tmp.flux, 'k-')
	plt.plot(model.wave, model.flux, 'r-')
	plt.show()
	plt.close()

	data.mask_custom(mask)

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
		plt.show()
		plt.close()

	return mask, pwv, float(airmass)

