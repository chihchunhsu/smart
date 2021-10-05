import copy
import numpy as np
import matplotlib.pyplot as plt
import smart

def get_bad_pixel(sp):
	"""
	Generate a bad pixel mask for the sciecne spectrum.
	"""

	pixel = np.arange(len(sp.wave))

	bad_pixel1 = pixel[np.isnan(sp.flux)].tolist()
	bad_pixel2 = pixel[np.isnan(sp.noise)].tolist()
	bax_pixel  = list(set(bad_pixel1+bad_pixel2))

	bax_pixel.sort()

	return bax_pixel

def generate_telluric_mask(name, order, path, pixel_start=10, pixel_end=None, sigma=3.0, guess_pwv=True, diagnostic=True):
	"""
	Generate a bad pixel mask for the telluric spectrum after the first wavelength calibration.

	Input
	-----


	Returns
	-------
	mask	:	 
	"""

	# read in the calibrated data
	tell_data  = smart.Spectrum(name=name, order=order, path=path)

	# edge pixel masking
	if pixel_end is None:
		# old NIRSPEC
		if len(tell_data.oriWave) == 1024:
			pixel_end = -30
		# upgraded NIRSPEC
		else:
			pixel_end = -60

	pixel  = np.arange(len(tell_data.wave))
	pixel  = pixel[pixel_start:pixel_end]
	tell_data.wave = tell_data.wave[pixel_start:pixel_end]
	tell_data.flux = tell_data.flux[pixel_start:pixel_end]
	tell_data.noise = tell_data.noise[pixel_start:pixel_end]

	# determine lsf, airmass and pwv
	vbroad     = (299792.458)*np.mean(np.diff(tell_data.wave))/np.mean(tell_data.wave)
	alpha      = 1.0
	# use the airmass in the header
	airmass    = float(round(tell_data.header['AIRMASS']*2)/2)
	if airmass > 3.0: airmass = 3.0
	airmass    = str(airmass)
	pwv        = tell_data.header['PWV']

	wavelow, wavehigh = tell_data.wave[0]-10, tell_data.wave[-1]+10
	model_tmp  = smart.getTelluric(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
	tell_data  = smart.continuumTelluric(data=tell_data, model=model_tmp)

	tell_data_original = copy.deepcopy(tell_data)
	pixel_all  = np.arange(len(tell_data_original.oriWave))

	tell_model = smart.convolveTelluric(vbroad, tell_data, alpha=alpha, airmass=airmass, pwv=pwv)

	mask   = pixel[abs(tell_data.flux-tell_model.flux)>sigma*np.std(abs(tell_data.flux-tell_model.flux))]
	select = abs(tell_data.flux-tell_model.flux)<sigma*np.std(abs(tell_data.flux-tell_model.flux))

	if diagnostic:
		plt.figure(figsize=(10,6))
		plt.plot(tell_data_original.wave, tell_data_original.flux, color='k', alpha=0.5)
		plt.plot(tell_data.wave[select], tell_data.flux[select], color='darkblue', label='telluric data')
		plt.plot(tell_data.wave[select], tell_data.noise[select], color='k', alpha=0.5)
		plt.plot(tell_model.wave[select], tell_model.flux[select], color='r', label='telluric model', alpha=0.5)
		plt.legend(loc=3, fontsize=25)
		plt.xlabel('Wavelength ($\AA$)', fontsize=30)
		plt.ylabel('Flux (normalized)', fontsize=30)
		plt.xticks(fontsize=25)
		plt.yticks(fontsize=25)
		plt.minorticks_on()
		plt.xlim(tell_model.wave[0]-10, tell_model.wave[-1]+10)
		plt.ylim(-0.1, 1.4)
		plt.tight_layout()
		plt.savefig(path + '/telluric_mask.pdf')
		#plt.show()
		plt.close()

		# use chi2 to estimate pwv
	if guess_pwv:
		pwv_list = ['0.5', '1.0', '1.5', '2.5', '3.5', '5.0']
		pwv_chi2 = []
		for pwv in pwv_list:
			data_tmp       = copy.deepcopy(tell_data)

			model_tmp      = smart.getTelluric(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
			data_tmp       = smart.continuumTelluric(data=data_tmp, model=model_tmp)
			model_tmp      = smart.convolveTelluric(vbroad, data_tmp, alpha=alpha, airmass=airmass, pwv=pwv)

			pwv_chi2.append(smart.chisquare(data_tmp, model_tmp))
			pwv_chi2_array = np.array(pwv_chi2)

			pwv_min_index = np.where(pwv_chi2_array == np.min(pwv_chi2_array))[0][0]
			pwv           = pwv_list[pwv_min_index]

		if diagnostic:
			plt.plot(pwv_list, pwv_chi2, label='best pwv = {} mm'.format(pwv))
			plt.xlabel('pwv (mm)', fontsize=20)
			plt.ylabel('$\chi^2$', fontsize=20)
			plt.xticks(fontsize=20)
			plt.yticks(fontsize=20)
			plt.legend(fontsize=20)
			plt.tight_layout()
			plt.savefig(path + '/telluric_pwv_chi2.pdf')
			#plt.show()
			plt.close()
			#print('Determine pwv = {} mm'.format(pwv))

		# masking using the new pwv
		model_tmp  = smart.getTelluric(wavelow=tell_data.wave[0]-10, wavehigh=tell_data.wave[-1]+10, airmass=airmass, pwv=pwv)
		tell_data  = smart.continuumTelluric(data=tell_data, model=model_tmp)
	
		tell_model = smart.convolveTelluric(vbroad, tell_data, alpha=alpha, airmass=airmass, pwv=pwv)
	
		mask   = pixel[abs(tell_data.flux-tell_model.flux)>sigma*np.std(abs(tell_data.flux-tell_model.flux))]
		select = abs(tell_data.flux-tell_model.flux)<sigma*np.std(abs(tell_data.flux-tell_model.flux))
	
		if diagnostic:
			plt.figure(figsize=(10,6))
			plt.plot(tell_data_original.wave, tell_data_original.flux, color='k', alpha=0.5)
			plt.plot(tell_data.wave[select], tell_data.flux[select], color='darkblue', label='telluric data')
			plt.plot(tell_data.wave[select], tell_data.noise[select], color='k', alpha=0.5)
			plt.plot(tell_model.wave[select], tell_model.flux[select], color='r', label='telluric model', alpha=0.5)
			plt.legend(loc=3, fontsize=25)
			plt.xlabel('Wavelength ($\AA$)', fontsize=30)
			plt.ylabel('Flux (normalized)', fontsize=30)
			plt.xticks(fontsize=20)
			plt.yticks(fontsize=20)
			plt.minorticks_on()
			plt.xlim(tell_model.wave[0]-10, tell_model.wave[-1]+10)
			plt.ylim(-0.1, 1.4)
			plt.tight_layout()
			plt.savefig(path + '/telluric_mask.pdf')
			#plt.show()
			plt.close()

	mask0 = pixel_all[0:pixel_start]
	mask1 = pixel_all[pixel_end:-1]
	mask0 = mask0.tolist()
	mask1 = mask1.tolist()
	mask  = mask.tolist()

	#mask_final = mask
	mask_final = mask0 + mask + mask1

	if guess_pwv:
		return mask_final, pwv
	else:
		return mask_final





