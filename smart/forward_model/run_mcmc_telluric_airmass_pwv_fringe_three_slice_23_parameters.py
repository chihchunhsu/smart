import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from astropy.io import fits
import emcee
import tellurics
import smart
from multiprocessing import Pool
import corner
import os
import sys
import time
import copy
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run the forward-modeling routine for telluric files",\
	usage="run_mcmc_telluric.py order date_obs tell_data_name tell_path save_to_path")

parser.add_argument("order",metavar='o',type=int,
    default=None, help="order", nargs="+")

parser.add_argument("date_obs",type=str,
    default=None, help="source name", nargs="+")

parser.add_argument("tell_data_name",type=str,
    default=None, help="telluric data name", nargs="+")

parser.add_argument("tell_path",type=str,
    default=None, help="telluric data path", nargs="+")

parser.add_argument("save_to_path",type=str,
    default=None, help="output path", nargs="+")

#parser.add_argument("-ndim",type=int,
#    default=5, help="number of dimension; default 17")

parser.add_argument("-nwalkers",type=int,
    default=50, help="number of walkers of MCMC; default 50")

parser.add_argument("-step",type=int,
    default=400, help="number of steps of MCMC; default 400")

parser.add_argument("-burn",type=int,
    default=300, help="burn of MCMC; default 300")

parser.add_argument("-moves",type=float,
    default=2.0, help="moves of MCMC; default 2.0")

parser.add_argument("-priors",type=dict,
    default=None, help="priors of MCMC; default None")

parser.add_argument("-pixel_start",type=int,
    default=10, help="starting pixel index for the science data; default 10")

parser.add_argument("-pixel_end",type=int,
    default=-40, help="ending pixel index for the science data; default -40")

parser.add_argument("-applymask",type=bool,
    default=False, help="apply a simple mask based on the STD of the average flux; default is False")

parser.add_argument("-save",type=bool,
    default=True, help="save the fitted LSF and telluric alpha to the fits file header; default is True")

args = parser.parse_args()

######################################################################################################


order                  = int(args.order[0])
date_obs               = str(args.date_obs[0])
tell_data_name         = str(args.tell_data_name[0])
tell_path              = str(args.tell_path[0])
save_to_path           = str(args.save_to_path[0])
#ndim, nwalkers, step   = int(args.ndim), int(args.nwalkers), int(args.step)
nwalkers, step         = int(args.nwalkers), int(args.step)
burn                   = int(args.burn)
moves                  = float(args.moves)
priors                 = args.priors
applymask              = args.applymask
pixel_start, pixel_end = int(args.pixel_start), int(args.pixel_end)
save                   = args.save

#lines                  = open(save_to_path+'/mcmc_parameters.txt').read().splitlines()
#custom_mask            = json.loads(lines[3].split('custom_mask')[1])
custom_mask = [] # test

if order == 35: applymask = True

tell_data_name2 = tell_data_name + '_calibrated'
tell_sp         = smart.Spectrum(name=tell_data_name2, order=order, path=tell_path, applymask=applymask)

# MJD for logging
# upgraded NIRSPEC
if len(tell_sp.oriWave) == 2048:
	mjd = tell_sp.header['MJD']
# old NIRSPEC
else:
	mjd = tell_sp.header['MJD-OBS']

###########################################################################################################
"""
MCMC routine for telluric standard stars to obtain the LSF and alpha. This function utilizes the emcee package.

Parameters
----------
tell_sp 	:	Spectrum object
				telluric spectrum
nwalkers 	:	int
				number of walkers. default is 30.
step 		:	int
				number of steps. default is 400
burn		:	int
			burn in mcmc to compute the best parameters. default is 300.
priors 		: 	dic
				A prior dictionary that specifies the range of the priors.
				Keys 	'lsf_min'  , 'lsf_max'  : LSF min/max in km/s
						'alpha_min', 'alpha_max': alpha min/max
						'A_min'    , 'A_max'	: flux offset in cnts/s
						'B_min'    , 'B_max'	: wave offset in Angstroms
				If there is no input priors dictionary, a wide range of priors will be adopted.
moves		:	float
				the stretch scale parameter. default is 2.0 (same as emcee).
save 		:	boolean
				save the modeled lsf and alpha in the header. default is True.
save_to_path: 	str
				the path to save the mcmc outputs.

"""

## Initial parameters setup
tell_data_name       = tell_sp.name
tell_path            = tell_sp.path
order                = tell_sp.order
ndim                 = 23
#applymask            = False
#pixel_start          = 10
#pixel_end            = -30

## apply a custom mask
print('masking pixels:', custom_mask)
tell_sp.mask_custom(custom_mask=custom_mask)

## add a pixel label for plotting
length1     = len(tell_sp.oriWave)
pixel       = np.delete(np.arange(length1),tell_sp.mask)
pixel       = pixel[pixel_start:pixel_end]

if save_to_path is None:
	save_to_path = './mcmc'

if save_to_path is not None:
	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)

if priors is None:
	#if tell_sp.header['AIRMASS'] < 3.0:
	#	airmass_0 = tell_sp.header['AIRMASS']
	#else:
	#	airmass_0 = 2.9

	### estimating the pwv parameter
	#pwv_list = [0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0]
	#pwv_chi2 = []
	#for pwv in pwv_list:
	#	data_tmp       = copy.deepcopy(tell_sp)
	#
	#	data_tmp.flux  = data_tmp.flux[pixel_start:pixel_end]
	#	data_tmp.wave  = data_tmp.wave[pixel_start:pixel_end]
	#	data_tmp.noise = data_tmp.noise[pixel_start:pixel_end]
	#	model_tmp      = tellurics.makeTelluricModel(lsf=5.0, airmass=round(airmass_0*2)/2, pwv=pwv, 
	#		flux_offset=0, wave_offset=0, data=data_tmp)
	#
	#	pwv_chi2.append(smart.chisquare(data_tmp, model_tmp))
	#
	## find the pwv with minimum chisquare
	#pwv_chi2_array = np.array(pwv_chi2)
	#
	#plt.plot(pwv_list, pwv_chi2)
	#plt.xlabel('pwv (mm)')
	#plt.ylabel('$\chi^2$')
	#plt.minorticks_on()
	#plt.tight_layout()
	#plt.savefig(save_to_path+'/pwv_chisquare_comparison.png', bbox_inches='tight')
	##plt.show()
	#plt.close()
	##sys.exit()

	#pwv_min_index = np.where(pwv_chi2_array == np.min(pwv_chi2_array))[0][0]
	#pwv_0         = pwv_list[pwv_min_index]

	#priors      =	{	'lsf_min':4.9  		,  'lsf_max':5.0,
	#					'airmass_min':1.02  ,  'airmass_max':1.03,		
	#					'pwv_min':1.48      ,  'pwv_max':1.49,
	#					'A_min':0.0297 		,  'A_max':0.0298,
	#					'B_min':0.0135  	,  'B_max':0.0136   
	#				}

	priors      =	{	'lsf_min':4.0  		,  'lsf_max':5.5,
						'airmass_min':1.00  ,  'airmass_max':1.50,		
						'pwv_min':0.5       ,  'pwv_max':5.0,
						'A_min':-1.0 		,  'A_max':1.0,
						'B_min':-0.02  		,  'B_max':0.02   
					}


	priors['a1_1'], priors['k1_1'], priors['p1_1'], priors['a2_1'], priors['k2_1'], priors['p2_1'] = 0.0174107, 2.10231814, -3.14150313, 0.00484561, 0.86503406, 3.14156183
	priors['a1_2'], priors['k1_2'], priors['p1_2'], priors['a2_2'], priors['k2_2'], priors['p2_2'] = 0.0144445, 2.0740762, -0.00275613, 0.00224659, 0.84935461, -0.00915833
	priors['a1_3'], priors['k1_3'], priors['p1_3'], priors['a2_3'], priors['k2_3'], priors['p2_3'] = 0.01452451, 2.07021129, 0.82740255, 0.00826017, 0.82431895, -3.14154979

	fringe_parameters = [	'a1_1', 'k1_1', 'p1_1', 'a2_1', 'k2_1', 'p2_1', 
							'a1_2', 'k1_2', 'p1_2', 'a2_2', 'k2_2', 'p2_2', 
							'a1_3', 'k1_3', 'p1_3', 'a2_3', 'k2_3', 'p2_3']

	for key in fringe_parameters:
		if 'a' or 'k' in key:
			priors[key+'_min'] = priors[key] * 0.999
			priors[key+'_max'] = priors[key] * 1.001
		elif 'p' in key:
			if priors[key] < -3.14:
				priors[key+'_min'] = max(priors[key] * 1.01, -np.pi)
				priors[key+'_max'] = priors[key] * 0.99
			elif priors[key] > 3.14:
				priors[key+'_min'] = priors[key] * 0.99
				priors[key+'_max'] = max(priors[key] * 1.01, np.pi)
			else:
				priors[key+'_min'] = min(priors[key] * 0.99, priors[key] * 1.01)
				priors[key+'_max'] = max(priors[key] * 0.99, priors[key] * 1.01)

		#if 'a' in key:
		#	priors[key+'_min'] = 0.00
		#	priors[key+'_max'] = 0.05
		#elif 'k1' in key:
		#	priors[key+'_min'] = 2.00
		#	priors[key+'_max'] = 2.20
		#elif 'k2' in key:
		#	priors[key+'_min'] = 0.75
		#	priors[key+'_max'] = 0.85
		#elif 'p' in key:
		#	priors[key+'_min'] = -np.pi
		#	priors[key+'_max'] = np.pi


tell_sp.wave    = tell_sp.wave[pixel_start:pixel_end]
tell_sp.flux    = tell_sp.flux[pixel_start:pixel_end]
tell_sp.noise   = tell_sp.noise[pixel_start:pixel_end]

## remove nan values in the noise
mask    = np.isnan(tell_sp.noise)

## invert the boolean mask and select only the non-nan values
tell_sp.wave  = tell_sp.wave[np.invert(mask)]
tell_sp.flux  = tell_sp.flux[np.invert(mask)]
tell_sp.noise = tell_sp.noise[np.invert(mask)]
pixel         = pixel[np.invert(mask)]
"""
## perform outlier rejection based on the lowest chi-square grid model
data_tmp       = copy.deepcopy(tell_sp)

model_tmp      = tellurics.makeTelluricModel(lsf=5.0, airmass=round(airmass_0*2)/2, pwv=pwv_0, 
	flux_offset=0, wave_offset=0, data=data_tmp)

outlier_rejection = 3.

plt.figure(figsize=(16,6))
plt.plot(tell_sp.wave, tell_sp.flux, 'k-', alpha=0.5)
plt.plot(model_tmp.wave, model_tmp.flux, 'r-', alpha=0.5)
plt.plot(tell_sp.wave, tell_sp.flux-model_tmp.flux, 'k-', alpha=0.5)
plt.fill_between(tell_sp.wave, -outlier_rejection*tell_sp.noise, outlier_rejection*tell_sp.noise, color='C0', alpha=0.5)
plt.savefig(save_to_path+'/mask_comparison_before.png', bbox_inches='tight')
#plt.show()
plt.close()

select         = np.where( abs(data_tmp.flux-model_tmp.flux) < outlier_rejection * data_tmp.noise)
tell_sp.wave  = tell_sp.wave[select]
tell_sp.flux  = tell_sp.flux[select]
tell_sp.noise = tell_sp.noise[select]
pixel         = pixel[select]

plt.figure(figsize=(16,6))
plt.plot(tell_sp.wave, tell_sp.flux, 'k-', alpha=0.5)
plt.plot(model_tmp.wave, model_tmp.flux, 'r-', alpha=0.5)
plt.plot(tell_sp.wave, tell_sp.flux-model_tmp.flux[select], 'k-', alpha=0.5)
plt.fill_between(tell_sp.wave, -outlier_rejection*tell_sp.noise, outlier_rejection*tell_sp.noise, color='C0', alpha=0.5)
plt.savefig(save_to_path+'/mask_comparison_after.png', bbox_inches='tight')
#plt.show()
plt.close()
#sys.exit()
"""
###########################################################################################################################

data = copy.deepcopy(tell_sp)


## log file
log_path = save_to_path + '/mcmc_parameters.txt'

file_log = open(log_path,"a+")
#file_log = open(log_path,"w+")
#file_log.write("tell_path {} \n".format(tell_path))
#file_log.write("tell_name {} \n".format(tell_data_name))
#file_log.write("order {} \n".format(order))
file_log.write("ndim {} \n".format(ndim))
file_log.write("nwalkers {} \n".format(nwalkers))
file_log.write("step {} \n".format(step))
file_log.write("pixel_start {} \n".format(pixel_start))
file_log.write("pixel_end {} \n".format(pixel_end))
file_log.write("moves {} \n".format(moves))
file_log.close()

## MCMC for the parameters LSF, alpha, and a nuisance parameter for flux offset
def lnlike(theta, data=data):
	"""
	Log-likelihood, computed from chi-squared.

	Parameters
	----------
	theta
	data

	Returns
	-------
	-0.5 * chi-square + sum of the log of the noise
	"""
	## Parameters MCMC

	lsf, airmass, pwv, A, B, \
	a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, \
    a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, \
    a1_3, k1_3, p1_3, a2_3, k2_3, p2_3 = theta

	model = tellurics.makeTelluricModelFringe(	lsf, airmass, pwv, A, B, 
												a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, 
												a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, 
												a1_3, k1_3, p1_3, a2_3, k2_3, p2_3,
												data=data, deg=2, niter=None)

	chisquare = smart.chisquare(data, model)

	return -0.5 * (chisquare + np.sum(np.log(2*np.pi*data.noise**2)))

def lnprior(theta):
	"""
	Specifies a flat prior
	"""
	## Parameters for theta
	lsf, airmass, pwv, A, B, \
	a1_1, k1_1, p1_1, a2_1, k2_1, p2_1, \
    a1_2, k1_2, p1_2, a2_2, k2_2, p2_2, \
    a1_3, k1_3, p1_3, a2_3, k2_3, p2_3 = theta

	limits =  { 'lsf_min':2.0  		,  'lsf_max':10.0,
				'airmass_min':1.0   ,  'airmass_max':3.0,
				'pwv_min':0.50 		,	'pwv_max':20.0,
				'A_min':-500.0 		,  'A_max':500.0,
				'B_min':-0.04  	    ,  'B_max':0.04    }

	for number in ['1', '2', '3']:
		# amplitude
		limits[f'a1_{number}_min'], limits[f'a1_{number}_max'] = 0.010, 0.020
		limits[f'a2_{number}_min'], limits[f'a2_{number}_max'] = 0.002, 0.012

		# wave number
		limits[f'k1_{number}_min'], limits[f'k1_{number}_max'] = 2.02, 2.12
		limits[f'k2_{number}_min'], limits[f'k2_{number}_max'] = 0.80, 0.89

		# phase; should be close to pi, 0, or -pi
		limits[f'p1_{number}_min'], limits[f'p1_{number}_max'] = -np.pi, np.pi
		limits[f'p2_{number}_min'], limits[f'p2_{number}_max'] = -np.pi, np.pi

	if  limits['lsf_min']     < lsf     < limits['lsf_max'] \
	and limits['airmass_min'] < airmass < limits['airmass_max']\
	and limits['pwv_min']     < pwv     < limits['pwv_max']\
	and limits['A_min']       < A       < limits['A_max']\
	and limits['B_min']       < B       < limits['B_max']\
	and limits['a1_1_min']    < a1_1    < limits['a1_1_max']\
	and limits['k1_1_min']    < k1_1    < limits['k1_1_max']\
	and limits['p1_1_min']    < p1_1    < limits['p1_1_max']\
	and limits['a2_1_min']    < a2_1    < limits['a2_1_max']\
	and limits['k2_1_min']    < k2_1    < limits['k2_1_max']\
	and limits['p2_1_min']    < p2_1    < limits['p2_1_max']\
	and limits['a1_2_min']    < a1_2    < limits['a1_2_max']\
	and limits['k1_2_min']    < k1_2    < limits['k1_2_max']\
	and limits['p1_2_min']    < p1_2    < limits['p1_2_max']\
	and limits['a2_2_min']    < a2_2    < limits['a2_2_max']\
	and limits['k2_2_min']    < k2_2    < limits['k2_2_max']\
	and limits['p2_2_min']    < p2_2    < limits['p2_2_max']\
	and limits['a1_3_min']    < a1_3    < limits['a1_3_max']\
	and limits['k1_3_min']    < k1_3    < limits['k1_3_max']\
	and limits['p1_3_min']    < p1_3    < limits['p1_3_max']\
	and limits['a2_3_min']    < a2_3    < limits['a2_3_max']\
	and limits['k2_3_min']    < k2_3    < limits['k2_3_max']\
	and limits['p2_3_min']    < p2_3    < limits['p2_3_max']:
		return 0.0

	return -np.inf

def lnprob(theta, data):
	lnp = lnprior(theta)

	if not np.isfinite(lnp):
		return -np.inf
	return lnp + lnlike(theta, data)

pos = [np.array([priors['lsf_min']       + (priors['lsf_max']        - priors['lsf_min'] )  * np.random.uniform(),
				 priors['airmass_min']   + (priors['airmass_max']    - priors['airmass_min'] )  * np.random.uniform(), 
				 priors['pwv_min']       + (priors['pwv_max']        - priors['pwv_min'] )  * np.random.uniform(), 
				 priors['A_min']         + (priors['A_max']          - priors['A_min'])     * np.random.uniform(),
				 priors['B_min']         + (priors['B_max']          - priors['B_min'])     * np.random.uniform(),
				 priors['a1_1_min']      + (priors['a1_1_max']       - priors['a1_1_min'])  * np.random.uniform(),
				 priors['k1_1_min']      + (priors['k1_1_max']       - priors['k1_1_min'])  * np.random.uniform(),
				 priors['p1_1_min']      + (priors['p1_1_max']       - priors['p1_1_min'])  * np.random.uniform(),
				 priors['a2_1_min']      + (priors['a2_1_max']       - priors['a2_1_min'])  * np.random.uniform(),
				 priors['k2_1_min']      + (priors['k2_1_max']       - priors['k2_1_min'])  * np.random.uniform(),
				 priors['p2_1_min']      + (priors['p2_1_max']       - priors['p2_1_min'])  * np.random.uniform(),
				 priors['a1_2_min']      + (priors['a1_2_max']       - priors['a1_2_min'])  * np.random.uniform(),
				 priors['k1_2_min']      + (priors['k1_2_max']       - priors['k1_2_min'])  * np.random.uniform(),
				 priors['p1_2_min']      + (priors['p1_2_max']       - priors['p1_2_min'])  * np.random.uniform(),
				 priors['a2_2_min']      + (priors['a2_2_max']       - priors['a2_2_min'])  * np.random.uniform(),
				 priors['k2_2_min']      + (priors['k2_2_max']       - priors['k2_2_min'])  * np.random.uniform(),
				 priors['p2_2_min']      + (priors['p2_2_max']       - priors['p2_2_min'])  * np.random.uniform(),
				 priors['a1_3_min']      + (priors['a1_3_max']       - priors['a1_3_min'])  * np.random.uniform(),
				 priors['k1_3_min']      + (priors['k1_3_max']       - priors['k1_3_min'])  * np.random.uniform(),
				 priors['p1_3_min']      + (priors['p1_3_max']       - priors['p1_3_min'])  * np.random.uniform(),
				 priors['a2_3_min']      + (priors['a2_3_max']       - priors['a2_3_min'])  * np.random.uniform(),
				 priors['k2_3_min']      + (priors['k2_3_max']       - priors['k2_3_min'])  * np.random.uniform(),
				 priors['p2_3_min']      + (priors['p2_3_max']       - priors['p2_3_min'])  * np.random.uniform()]) for i in range(nwalkers)]

with Pool() as pool:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves, pool=pool, moves=emcee.moves.KDEMove())
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves, pool=pool, moves=emcee.moves.StretchMove())
	time1 = time.time()
	sampler.run_mcmc(pos, step, progress=True)
	time2 = time.time()
	print('total time: ',(time2-time1)/60,' min.')
	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
	print(sampler.acceptance_fraction)
	autocorr_time = sampler.get_autocorr_time(discard=burn, quiet=True)
	print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(autocorr_time)))
	print(autocorr_time)

np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])
	
samples = sampler.chain[:, :, :].reshape((-1, ndim))

np.save(save_to_path + '/samples', samples)

# create walker plots
sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
samples = np.load(save_to_path + '/samples.npy')

ylabels = ["$\Delta \\nu_{inst}$ (km/s)", "airmass", "pwv (mm)", "$F_{\lambda}$ offset", "$\lambda$ offset ($\AA$)",
			'$a_{1, 1}$', '$k_{1, 1}(\AA^{-1})$', '$p{1, 1}$(rad)', '$a_{2, 1}$', '$k_{2, 1}(\AA^{-1})$', '$p_{2, 1}$(rad)', 
			'$a_{1, 2}$', '$k_{1, 2}(\AA^{-1})$', '$p{1, 2}$(rad)', '$a_{2, 2}$', '$k_{2, 2}(\AA^{-1})$', '$p_{2, 2}$(rad)', 
			'$a_{1, 3}$', '$k_{1, 3}(\AA^{-1})$', '$p{1, 3}$(rad)', '$a_{2, 3}$', '$k_{2, 3}(\AA^{-1})$', '$p_{2, 3}$(rad)']

## create walker plots
plt.rc('font', family='sans-serif')
plt.tick_params(labelsize=30)
fig = plt.figure(tight_layout=True, figsize=(4, ndim))
gs = gridspec.GridSpec(ndim, 1)
gs.update(hspace=0.1)

for i in range(ndim):
	ax = fig.add_subplot(gs[i, :])
	for j in range(nwalkers):
		ax.plot(np.arange(1,int(step+1)), sampler_chain[j,:,i],'k',alpha=0.2)
	ax.set_ylabel(ylabels[i])
fig.align_labels()
plt.minorticks_on()
plt.xlabel('nstep')
plt.savefig(save_to_path+'/walker.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# create array triangle plots
triangle_samples = sampler_chain[:, burn:, :].reshape((-1, ndim))
#print(triangle_samples.shape)

# create the final spectra comparison
lsf_mcmc, airmass_mcmc, pwv_mcmc, A_mcmc, B_mcmc, \
	a1_1_mcmc, k1_1_mcmc, p1_1_mcmc, a2_1_mcmc, k2_1_mcmc, p2_1_mcmc, \
	a1_2_mcmc, k1_2_mcmc, p1_2_mcmc, a2_2_mcmc, k2_2_mcmc, p2_2_mcmc, \
	a1_3_mcmc, k1_3_mcmc, p1_3_mcmc, a2_3_mcmc, k2_3_mcmc, p2_3_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
	zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

# add the summary to the txt file
file_log = open(log_path,"a")
file_log.write("*** Below is the summary *** \n")
file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
file_log.write("lsf_mcmc {} km/s\n".format(str(lsf_mcmc)))
file_log.write("airmass_mcmc {} km/s\n".format(str(airmass_mcmc)))
file_log.write("pwv_mcmc {} km/s\n".format(str(pwv_mcmc)))
file_log.write("A_mcmc {}\n".format(str(A_mcmc)))
file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
file_log.write("a1_1_mcmc {}\n".format(str(a1_1_mcmc))) 
file_log.write("k1_1_mcmc {}\n".format(str(k1_1_mcmc))) 
file_log.write("p1_1_mcmc {}\n".format(str(p1_1_mcmc))) 
file_log.write("a2_1_mcmc {}\n".format(str(a2_1_mcmc))) 
file_log.write("k2_1_mcmc {}\n".format(str(k2_1_mcmc))) 
file_log.write("p2_1_mcmc {}\n".format(str(p2_1_mcmc))) 
file_log.write("a1_2_mcmc {}\n".format(str(a1_2_mcmc))) 
file_log.write("k1_2_mcmc {}\n".format(str(k1_2_mcmc))) 
file_log.write("p1_2_mcmc {}\n".format(str(p1_2_mcmc))) 
file_log.write("a2_2_mcmc {}\n".format(str(a2_2_mcmc))) 
file_log.write("k2_2_mcmc {}\n".format(str(k2_2_mcmc))) 
file_log.write("p2_2_mcmc {}\n".format(str(p2_2_mcmc))) 
file_log.write("a1_3_mcmc {}\n".format(str(a1_3_mcmc))) 
file_log.write("k1_3_mcmc {}\n".format(str(k1_3_mcmc))) 
file_log.write("p1_3_mcmc {}\n".format(str(p1_3_mcmc))) 
file_log.write("a2_3_mcmc {}\n".format(str(a2_3_mcmc))) 
file_log.write("k2_3_mcmc {}\n".format(str(k2_3_mcmc))) 
file_log.write("p2_3_mcmc {}\n".format(str(p2_3_mcmc)))
file_log.close()

print(lsf_mcmc, airmass_mcmc, pwv_mcmc, A_mcmc, B_mcmc, \
	a1_1_mcmc, k1_1_mcmc, p1_1_mcmc, a2_1_mcmc, k2_1_mcmc, p2_1_mcmc, \
	a1_2_mcmc, k1_2_mcmc, p1_2_mcmc, a2_2_mcmc, k2_2_mcmc, p2_2_mcmc, \
	a1_3_mcmc, k1_3_mcmc, p1_3_mcmc, a2_3_mcmc, k2_3_mcmc, p2_3_mcmc)

if '_' in tell_sp.name:
	tell_data_name = tell_sp.name.split('_')[0]

## triangular plots
plt.rc('font', family='sans-serif')
fig = corner.corner(triangle_samples, 
	labels=ylabels,
	truths=[lsf_mcmc[0],
	airmass_mcmc[0],
	pwv_mcmc[0], 
	A_mcmc[0],
	B_mcmc[0],
	a1_1_mcmc[0],
	k1_1_mcmc[0],
	p1_1_mcmc[0],
	a2_1_mcmc[0],
	k2_1_mcmc[0],
	p2_1_mcmc[0],
	a1_2_mcmc[0],
	k1_2_mcmc[0],
	p1_2_mcmc[0],
	a2_2_mcmc[0],
	k2_2_mcmc[0],
	p2_2_mcmc[0],
	a1_3_mcmc[0],
	k1_3_mcmc[0],
	p1_3_mcmc[0],
	a2_3_mcmc[0],
	k2_3_mcmc[0],
	p2_3_mcmc[0]],
	quantiles=[0.16, 0.84],
	label_kwargs={"fontsize": 20})
plt.minorticks_on()
fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

data2               = copy.deepcopy(data)
data2.wave          = data2.wave + B_mcmc[0]
telluric_model      = tellurics.convolveTelluric(lsf_mcmc[0], airmass_mcmc[0], pwv_mcmc[0], data)
model, pcont        = smart.continuum(data=data, mdl=telluric_model, deg=2, tell=True)
model.flux         += A_mcmc[0]

model = tellurics.makeTelluricModelFringe(	lsf_mcmc[0], airmass_mcmc[0], pwv_mcmc[0], A_mcmc[0], B_mcmc[0], 
											a1_1_mcmc[0], k1_1_mcmc[0], p1_1_mcmc[0], a2_1_mcmc[0], k2_1_mcmc[0], p2_1_mcmc[0], 
											a1_2_mcmc[0], k1_2_mcmc[0], p1_2_mcmc[0], a2_2_mcmc[0], k2_2_mcmc[0], p2_2_mcmc[0], 
											a1_3_mcmc[0], k1_3_mcmc[0], p1_3_mcmc[0], a2_3_mcmc[0], k2_3_mcmc[0], p2_3_mcmc[0],
											data=data, deg=2, niter=None)

plt.tick_params(labelsize=20)
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(111)
ax1.plot(model.wave, model.flux, c='C3', ls='-', alpha=0.5)
ax1.plot(model.wave, np.polyval(pcont, model.wave) + A_mcmc[0], c='C1', ls='-', alpha=0.5)
ax1.plot(data.wave, data.flux, 'k-', alpha=0.5)
ax1.plot(data.wave, data.flux - model.flux,'k-', alpha=0.5)
ax1.minorticks_on()
plt.figtext(0.89,0.86,"{} O{}".format(tell_data_name, order),
	color='k',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.figtext(0.89,0.83,"LSF ${0}^{{+{1}}}_{{-{2}}} (km/s)/ airmass \, {3}^{{+{4}}}_{{-{5}}}/pwv {6}^{{+{7}}}_{{-{8}}}$ (mm)".format(\
	round(lsf_mcmc[0],2),
	round(lsf_mcmc[1],2),
	round(lsf_mcmc[2],2),
	round(airmass_mcmc[0],2),
	round(airmass_mcmc[1],2),
	round(airmass_mcmc[2],2),
	round(pwv_mcmc[0],2),
	round(pwv_mcmc[1],2),
	round(pwv_mcmc[2],2)),
	color='r',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.figtext(0.89,0.80,r"$\chi^2$ = {}, DOF = {}".format(\
	round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
	color='k',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.fill_between(data.wave, -data.noise, data.noise, alpha=0.5)
plt.tick_params(labelsize=15)
plt.ylabel('Flux (counts/s)',fontsize=15)
plt.xlabel('Wavelength ($\AA$)',fontsize=15)
plt.xlim(data.wave[0], data.wave[-1])

ax2 = ax1.twiny()
ax2.plot(pixel, data.flux, color='w', alpha=0)
ax2.set_xlabel('Pixel',fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xlim(pixel[0], pixel[-1])
ax2.minorticks_on()
	
plt.savefig(save_to_path+'/telluric_spectrum.png',dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

"""
# excel summary file
cat = pd.DataFrame(columns=[	'date_obs', 'tell_name', 'tell_path', 'snr_tell', 'tell_mask', 'order',
								'mjd_tell', 'ndim_tell', 'nwalker_tell', 'step_tell', 'burn_tell', 
								'pixel_start_tell', 'pixel_end_tell',
								'lsf_tell', 'lsf_tell_ue', 'lsf_tell_le', 
								'am_tell', 'am_tell_ue', 'am_tell_le', 
								'pwv_tell', 'pwv_tell_ue', 'pwv_tell_le',
								'A_tell', 'A_tell_ue', 'A_tell_le', 'B_tell', 'B_tell_ue', 'B_tell_le'])

snr_tell = np.nanmedian(tell_sp.flux/tell_sp.noise)

cat = cat.append({	'date_obs':date_obs, 'tell_name':tell_data_name, 'tell_path':tell_path, 'snr_tell':snr_tell,
					'tell_mask':custom_mask, 'order':order, 'mjd_tell':mjd, 
					'ndim_tell':ndim, 'nwalker_tell':nwalkers, 'step_tell':step, 'burn_tell':burn,
					'pixel_start_tell':pixel_start, 'pixel_end_tell':pixel_end,
					'lsf_tell':lsf_mcmc[0], 'lsf_tell_ue':lsf_mcmc[1], 'lsf_tell_le':lsf_mcmc[2], 
					'am_tell':airmass_mcmc[0], 'am_tell_ue':airmass_mcmc[1], 'am_tell_le':airmass_mcmc[2], 
					'pwv_tell':pwv_mcmc[0], 'pwv_tell_ue':pwv_mcmc[1], 'pwv_tell_le':pwv_mcmc[2],
					'A_tell':A_mcmc[0], 'A_tell_ue':A_mcmc[1], 'A_tell_le':A_mcmc[2], 
					'B_tell':B_mcmc[0], 'B_tell_ue':B_mcmc[1], 'B_tell_le':B_mcmc[2]}, ignore_index=True)

cat.to_excel(save_to_path + '/mcmc_summary.xlsx', index=False)


# save the best fit parameters in the fits header
if save is True:
	data_path = tell_sp.path + '/' + tell_sp.name + '_' + str(tell_sp.order) + '_all.fits'
	with fits.open(data_path) as hdulist:
		hdulist[0].header['LSF']          = lsf_mcmc[0]
		hdulist[0].header['AM_FIT']       = airmass_mcmc[0]
		hdulist[0].header['PWV_FIT']      = pwv_mcmc[0]
		try:
			hdulist.writeto(data_path, overwrite=True)
		except FileNotFoundError:
			hdulist.writeto(data_path)
"""