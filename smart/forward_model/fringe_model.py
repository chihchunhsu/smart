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
	
	best_frequency1 = f[(f>1.5) & (f<2.8)][np.argmax(pgram[(f>1.5) & (f<2.8)])]
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

def double_sine2(wave, a1, k1, p1, a2, k2, p2):
	"""
	Double sine function for the fringe pattern that assume wave numnber k times wavekength x with a fringe amplitdue a,
	as well a phase term for each sine function.

	The initial guess is determined from the best frequency.
	"""
	return (1 + a1**2 + 2 * a1*np.sin( k1*wave + p1 )) * ( 1 + a2**2 + 2 * a2*np.sin( k2*wave + p2 )) - 1


def fit_fringe_model_parameter(fringe_object, pixel_start, pixel_end):
	"""
	Get the best-fit parameters for a double-sine fringe model
	"""

	amp = max(tmp.flux)
	p0 = [amp, best_frequency1, 0.0, amp, best_frequency2, 0.0]
	bounds = ([0.0, 0.0, -np.pi, 0.0, 0.0, -np.pi], 
		[1.1*amp, 100*best_frequency1, np.pi, 1.1*amp, 100*best_frequency2, np.pi])
	popt, pcov = curve_fit(doub_sin, tmp.wave, tmp.flux, 
		maxfev=10000, p0=p0, bounds=bounds)

	pass

def double_sine_fringe(data, piecewise_fringe_model, teff, logg, vsini, rv, airmass, pwv, wave_offset, flux_offset, lsf, modelset, output_stellar_model=False):
	"""
	***CURRENTLY STILL UNDERDEVELOPEMENT*** 
	Make a peice-wise double sine fringe model for a stellar spectrum, with each fringe model having 6 parameters.
	"""

	# make a model without the fringe
	if not output_stellar_model:
		model_tmp = smart.makeModel(teff=teff, logg=logg, metal=0.0, 
			vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=wave_offset, flux_offset=flux_offset,
			lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=airmass, pwv=pwv, output_stellar_model=False)
	else:
		model_tmp, model_tmp_notell = smart.makeModel(teff=teff, logg=logg, metal=0.0, 
			vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=wave_offset, flux_offset=flux_offset,
			lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=airmass, pwv=pwv, output_stellar_model=True)


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
		if pixel_start == -300:
			best_frequency1, best_frequency2 = 1.65, 0.75
		#print(best_frequency1, best_frequency2)

		amp = max(tmp.flux)
		p0 = [amp, best_frequency1, 0.0, amp, best_frequency2, 0.0]
		bounds = ([0.0, 0.0, -np.pi, 0.0, 0.0, -np.pi], 
			[1.1*amp, 100*best_frequency1, np.pi, 1.1*amp, 100*best_frequency2, np.pi])

		#try:
		popt, pcov = curve_fit(double_sine2, tmp.wave, tmp.flux, maxfev=10000, p0=p0, bounds=bounds)

		# replace the model with the fringe pattern; note that this has to be the model wavelength at the current forward-modeling step before resampling
		#model.flux[(model.wave>residual.wave[pixel_start]) & (model.wave<residual.wave[pixel_end])] *= (1 + double_sine(model.wave[[(model.wave>residual.wave[pixel_start]) & (model.wave<residual.wave[pixel_end])]], *popt))

		#except:
		#	pass
		model_tmp.flux[pixel_start: pixel_end] = model_tmp.flux[pixel_start: pixel_end]*(1 + double_sine2(residual.wave[pixel_start: pixel_end], *popt))

		if output_stellar_model:
			model_tmp_notell.flux[pixel_start: pixel_end] = model_tmp_notell.flux[pixel_start: pixel_end]*(1 + double_sine2(residual.wave[pixel_start: pixel_end], *popt))


	if not output_stellar_model:
		return model_tmp
	else:
		return model_tmp, model_tmp_notell

