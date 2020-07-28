import smart
import numpy as np

def _residual(data,model):
	"""
	Return a residual flux array with the length of the data. (deprecated)
	"""
	residual = []
	# find the region where the model is in the range of the data
	data_model_range = np.where(np.logical_and(np.array(model.wave) >= data.wave[0], \
		np.array(model.wave) <= data.wave[-1]))[0]
	#residual = np.zeros(len(data_model_range))
	for i in data_model_range:
		model_wave = model.wave[i]
		j = np.isclose(np.array(data.wave), model_wave)
		if len(np.where(j)[0]) is 1:
			residual.append(float(data.flux[j] - model.flux[i]))
			#residual[i] = float(data.flux[j] - model.flux[i])
		else:
		# take the average of the wavelength if there are more than
		# one data point close to the model
			data_flux = np.average(data.flux[j])
			residual.append(float(data_flux - model.flux[i]))
			#residual[i] = float(data_flux - model.flux[i])

	residual_model = smart.Model()
	residual_model.wave = model.wave[data_model_range]
	residual_model.flux = np.asarray(residual)
	# reject fluxes larger than 5 sigmas
	#residual_model.flux = residual_model.flux[np.absolute(residual_model.flux)<5*np.std(residual_model.flux)]
	#residual_model.wave = residual_model.wave[np.absolute(residual_model.flux)<5*np.std(residual_model.flux)]
	return residual_model

def residual(data, model):
	"""
	Return a residual flux array with the length of the data.
	"""
	if np.array_equal(data.wave,model.wave):
		residual_model      = smart.Model()
		residual_model.flux = data.flux - model.flux
		residual_model.wave = data.wave

		return residual_model
	
	else:
		print("The wavelength arrays of the data and model are not equal.")

		return None

