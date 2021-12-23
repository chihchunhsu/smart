#########################################################
#
#	Fringe Model Functions
#
#########################################################
import sys, copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
import smart

def get_peak_fringe_frequency(fringe_object, pixel_start, pixel_end):
	"""
	Get the two peak frequencies for the fringe pattern.
	"""
	tmp = copy.deepcopy(fringe_object)

	tmp.flux = tmp.flux[pixel_start: pixel_end]
	tmp.wave = tmp.wave[pixel_start: pixel_end]

	f = np.linspace(0.01, 10.0, 10000)
	pgram = signal.lombscargle(tmp.wave, tmp.flux, f, normalize=True)
	
	best_frequency1 = f[np.argmax(pgram)]
	#print(best_frequency1)
	best_frequency2 = f[(f>0.5) & (f<1.3)][np.argmax(pgram[(f>0.5) & (f<1.3)])]
	#print(best_frequency2)
	
	return best_frequency1, best_frequency2

def double_sine(wave, a1, k1, a2, k2):
	"""
	Double sine function for the fringe pattern that assume wave numnber k times wavekength x with a fringe amplitdue a.

	The initial guess is determined from the best frequency.
	"""

	return (1 + a1**2 + 2 * a1 * np.sin( k1 * wave )) * ( 1 + a2**2 + 2 * a2 * np.sin( k2 * wave )) - 1


def double_sine_fringe(model, data, piecewise_fringe_model, teff, logg, vsini, rv, airmass, pwv, wave_offset, flux_offset, lsf, modelset):

	# make a model without the fringe
	model_tmp = smart.makeModel(teff=teff, logg=logg, metal=0.0, 
		vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=wave_offset, flux_offset=flux_offset,
		lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=airmass, pwv=pwv, output_stellar_model=False)

	# construct the fringe model
	residual      = copy.deepcopy(data)
	residual.flux = (data.flux - model_tmp.flux)/model_tmp.flux

	end = len(piecewise_fringe_model)-1

	for i in range(len(piecewise_fringe_model)-1):
		pixel_start, pixel_end = piecewise_fringe_model[i], piecewise_fringe_model[i+1]

		tmp = copy.deepcopy(residual)
		tmp.flux = tmp.flux[pixel_start: pixel_end]
		tmp.wave = tmp.wave[pixel_start: pixel_end]

		#best_frequency1, best_frequency2 = get_peak_fringe_frequency(tmp, pixel_start, pixel_end)
		best_frequency1, best_frequency2 = 2.10, 0.85

		#amp = max(tmp.flux)
		amp = 0.01
		p0  = [amp, best_frequency1, amp, best_frequency2]
		bounds = (	[0.0, 0.0, 0.0, 0.0], 
					[2.0*amp, 100*best_frequency1, 2.0*amp, 100*best_frequency2])

		try:
			popt, pcov = curve_fit(double_sine, tmp.wave, tmp.flux, maxfev=10000, p0=p0, bounds=bounds)

			# replace the model with the fringe pattern; note that this has to be the model wavelength at the current forward-modeling step before resampling
			model.flux[(model.wave>residual.wave[pixel_start]) & (model.wave<residual.wave[pixel_end])] *= (1 + double_sine(model.wave[[(model.wave>residual.wave[pixel_start]) & (model.wave<residual.wave[pixel_end])]], *popt))

		except:
			pass

	return model.flux

