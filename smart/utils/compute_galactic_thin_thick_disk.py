###################################################################################
# Compute the Galactic thin/thick disk population based on Bensby et al. (2003)   #
###################################################################################

import numpy as np

###################################################
# Parameters of Table 1 from Bensby et al. (2003) #
###################################################

# velocity dispersions are in km/s

params = 	{	
				'thin':	{	'X':0.94, 		'sigmaU':35,	'sigmaV':20, 'sigmaW':16, 'Vasym':-15} ,
				'thick':{	'X':0.06, 		'sigmaU':67, 	'sigmaV':38, 'sigmaW':35, 'Vasym':-46} ,
				'halo':	{	'X':0.0015, 	'sigmaU':160, 	'sigmaV':90, 'sigmaW':90, 'Vasym':-220}
			}

def compute_galactic_pop_f(U, V, W, pop):
	"""
	Equation 1 from Bensby et al. (2003) to compute the probability of thin, thick, or halo populations.
	
	Parameters
	----------
	U 		:	Galactic U velocity (km/s)
	V 		:	Galactic V velocity (km/s)
	W 		:	Galactic W velocity (km/s)
	pop 	:	populations, can be 'thin', 'thick', or 'halo'

	Returns
	-------
	prob 	:	the probability of thin, thick, or halo populations.

	"""

	sigmaU 	= params[pop]['sigmaU']
	sigmaV 	= params[pop]['sigmaV']
	sigmaW 	= params[pop]['sigmaW']
	Vasym	= params[pop]['Vasym']

	k = 1/( 2 * np.pi**(1.5) * sigmaU * sigmaV * sigmaW )

	factor 	= np.exp( - U**2 / (2 * sigmaU**2) - ( V - Vasym )**2 / (2 * sigmaV**2) - W**2 / (2 * sigmaW**2))

	return k * factor

def compute_galactic_pop_prob(U, V, W, pop='thick_thin'):
	"""
	Equation 3 from Bensby et al. (2003) to compute the probability of thick/thin disk probability.
	"""

	## compute the thick and thin disk population
	thick_f	= compute_galactic_pop_f(U, V, W, 'thick')
	thin_f 	= compute_galactic_pop_f(U, V, W, 'thin')

	if pop == 'thick_thin':
		# compute the thick/thin disk probability
		prob 	= ( params['thick']['X'] / params['thin']['X'] ) * ( thick_f / thin_f )
	elif pop == 'halo_thick':
		prob 	= ( params['halo']['X'] / params['thick']['X'] ) * ( compute_galactic_pop_f(U, V, W, 'halo') / thick_f )

	return prob