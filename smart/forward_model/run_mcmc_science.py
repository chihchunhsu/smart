import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.gridspec as gridspec
from astropy.io import fits
import emcee
#from schwimmbad import MPIPool
from multiprocessing import Pool, set_start_method
import smart
import model_fit
import mcmc_utils
import corner
import os
import sys
import time
import copy
import argparse
import json
import ast
import warnings
from datetime import date, datetime
warnings.filterwarnings("ignore")

##############################################################################################
## This is the script to make the code multiprocessing, using arcparse to pass the arguments
## The code is run with 8 parameters, including Teff, logg, RV, vsini, telluric alpha, and 
## nuisance parameters for wavelength, flux and noise.
##############################################################################################

parser = argparse.ArgumentParser(description="Run the forward-modeling routine for science files",
	usage="run_mcmc_science.py order date_obs sci_data_name tell_data_name data_path tell_path save_to_path lsf priors limits")

#parser.add_argument("source",metavar='src',type=str,
#   default=None, help="source name", nargs="+")

parser.add_argument("order",metavar='o',type=int,
    default=None, help="order", nargs="+")

parser.add_argument("date_obs",metavar='dobs',type=str,
    default=None, help="source name", nargs="+")

parser.add_argument("sci_data_name",metavar='sci',type=str,
    default=None, help="science data name", nargs="+")

parser.add_argument("tell_data_name",metavar='tell',type=str,
    default=None, help="telluric data name", nargs="+")

parser.add_argument("data_path",type=str,
    default=None, help="science data path", nargs="+")

parser.add_argument("tell_path",type=str,
    default=None, help="telluric data path", nargs="+")

parser.add_argument("save_to_path",type=str,
    default=None, help="output path", nargs="+")

parser.add_argument("lsf",type=float,
    default=None, help="line spread function", nargs="+")

parser.add_argument("-instrument",metavar='--inst',type=str,
    default='nirspec', help="spectrometer name of the instrument; default nirspec")

parser.add_argument("-outlier_rejection",metavar='--rej',type=float,
    default=3.0, help="outlier rejection based on the multiple of standard deviation of the residual; default 3.0")

parser.add_argument("-ndim",type=int,
    default=8, help="number of dimension; default 8")

parser.add_argument("-nwalkers",type=int,
    default=50, help="number of walkers of MCMC; default 50")

parser.add_argument("-step",type=int,
    default=600, help="number of steps of MCMC; default 600")

parser.add_argument("-burn",type=int,
    default=500, help="burn of MCMC; default 500")

parser.add_argument("-moves",type=float,
    default=2.0, help="moves of MCMC; default 2.0")

parser.add_argument("-pixel_start",type=int,
    default=10, help="starting pixel index for the science data; default 10")

parser.add_argument("-pixel_end",type=int,
    default=-40, help="ending pixel index for the science data; default -40")

#parser.add_argument("-pwv",type=float,
#    default=0.5, help="precipitable water vapor for telluric profile; default 0.5 mm")

#parser.add_argument("-alpha_tell",type=float,
#    default=1.0, help="telluric alpha; default 1.0")

parser.add_argument("-apply_sigma_mask",type=bool,
    default=False, help="apply a simple mask based on the STD of the average flux; default is False")

parser.add_argument("-plot_show",type=bool,
    default=False, help="show the MCMC plots; default is False")

parser.add_argument("-coadd",type=bool,
    default=False, help="coadd the spectra; default is False")

parser.add_argument("-coadd_sp_name",type=str,
    default=None, help="name of the coadded spectra")

parser.add_argument("-modelset",type=str,
    default='btsettl08', help="model set; default is btsettl08")

parser.add_argument("-final_mcmc", action='store_true', help="run final mcmc; default False")

parser.add_argument("-include_fringe_model", action='store_true', help="model the fringe pattern; default False")

args = parser.parse_args()

######################################################################################################

#source                 = str(args.source[0])
order                  = int(args.order[0])
instrument             = str(args.instrument)
date_obs               = str(args.date_obs[0])
sci_data_name          = str(args.sci_data_name[0])
data_path              = str(args.data_path[0])
tell_data_name         = str(args.tell_data_name[0])
tell_path              = str(args.tell_path[0])
save_to_path_base      = str(args.save_to_path[0])
lsf                    = float(args.lsf[0])
ndim, nwalkers, step   = int(args.ndim), int(args.nwalkers), int(args.step)
burn                   = int(args.burn)
moves                  = float(args.moves)
apply_sigma_mask              = args.apply_sigma_mask
pixel_start, pixel_end = int(args.pixel_start), int(args.pixel_end)
#pwv                    = float(args.pwv)
#alpha_tell             = float(args.alpha_tell[0])
plot_show              = args.plot_show
coadd                  = args.coadd
outlier_rejection      = float(args.outlier_rejection)
modelset               = str(args.modelset)
final_mcmc             = args.final_mcmc
include_fringe_model   = args.include_fringe_model

if final_mcmc:
	#save_to_path1  = save_to_path_base + '/init_mcmc'
	save_to_path   = save_to_path_base + '/final_mcmc'

else:
	save_to_path   = save_to_path_base + '/init_mcmc'

# date
today     = date.today()
now       = datetime.now()
dt_string = now.strftime("%H:%M:%S")	

#####################################

data        = smart.Spectrum(name=sci_data_name, order=order, path=data_path, apply_sigma_mask=apply_sigma_mask, instrument=instrument)

if instrument != 'hires':
	tell_data_name2 = tell_data_name + '_calibrated'
	tell_sp     = smart.Spectrum(name=tell_data_name2, order=data.order, path=tell_path, apply_sigma_mask=apply_sigma_mask, instrument=instrument)

	# TBD if the improved wavecal for KPIC data is possible
	if instrument != 'kpic':
		data.updateWaveSol(tell_sp)

# MJD for logging
# upgraded NIRSPEC
if instrument in ['nirspec', 'kpic']:
	if len(data.oriWave) == 2048:
		mjd = data.header['MJD']
	# old NIRSPEC
	else:
		mjd = data.header['MJD-OBS']

elif instrument == 'hires':
	mjd = data.header['MJD']

if coadd:
	sci_data_name2 = str(args.coadd_sp_name)
	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)
	data1       = copy.deepcopy(data)
	data2       = smart.Spectrum(name=sci_data_name2, order=order, path=data_path, apply_sigma_mask=apply_sigma_mask)
	data.coadd(data2, method='pixel')

	plt.figure(figsize=(16,6))
	plt.plot(np.arange(1024),data.flux,'k',
		label='coadd median S/N = {}'.format(np.median(data.flux/data.noise)),alpha=1)
	plt.plot(np.arange(1024),data1.flux,'C0',
		label='{} median S/N = {}'.format(sci_data_name,np.median(data1.flux/data1.noise)),alpha=0.5)
	plt.plot(np.arange(1024),data2.flux,'C1',
		label='{} median S/N = {}'.format(sci_data_name2,np.median(data2.flux/data2.noise)),alpha=0.5)
	plt.plot(np.arange(1024),data.noise,'k',alpha=0.5)
	plt.plot(np.arange(1024),data1.noise,'C0',alpha=0.5)
	plt.plot(np.arange(1024),data2.noise,'C1',alpha=0.5)
	plt.legend()
	plt.xlabel('pixel')
	plt.ylabel('cnts/s')
	plt.minorticks_on()
	plt.savefig(save_to_path+'/coadd_spectrum.png')
	#plt.show()
	plt.close()

sci_data  = data

if instrument != 'hires':
	tell_data = tell_sp 

"""
MCMC run for the science spectra. See the parameters in the makeModel function.

Parameters
----------

sci_data  	: 	sepctrum object
				science data

tell_data 	: 	spectrum object
				telluric data for calibrating the science spectra

priors   	: 	dic
				keys are teff_min, teff_max, logg_min, logg_max, vsini_min, vsini_max, rv_min, rv_max, 
				am_min, am_max, pwv_min, pwv_max, A_min, A_max, B_min, B_max

Optional Parameters
-------------------

limits 		:	dic
					mcmc limits with the same format as the input priors

ndim 		:	int
				mcmc dimension

nwalkers 	:	int
					number of walkers

	step 		: 	int
					number of steps

	burn 		:	int
					burn for the mcmc

	moves 		: 	float
					stretch parameter for the mcmc. The default is 2.0 based on the emcee package

	pixel_start	:	int
					starting pixel number for the spectra in the MCMC

	pixel_end	:	int
					ending pixel number for the spectra in the MCMC

	alpha_tell	:	float
					power of telluric spectra for estimating the line spread function of the NIRSPEC instrument

	modelset 	:	str
					'btsettl08' or 'phoenixaces' model sets

	save_to_path: 	str
					path to savr the MCMC output				

"""

if save_to_path is not None:
	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)
else:
	save_to_path = '.'

#if limits is None: limits = priors

data          = copy.deepcopy(sci_data)

if instrument != 'hires':
	tell_sp       = copy.deepcopy(tell_data)
	
	if instrument != 'kpic':
		data.updateWaveSol(tell_sp)

# barycentric corrction
#barycorr      = smart.barycorr(data.header).value
#print("barycorr:",barycorr)

## read the input custom mask and priors
lines          = open(save_to_path+'/mcmc_parameters.txt').read().splitlines()
if instrument in ['nirspec', 'kpic']:
	custom_mask    = json.loads(lines[5].split('custom_mask')[1])
	priors         = ast.literal_eval(lines[6].split('priors ')[1])
	if include_fringe_model:
		fringe_param   = ast.literal_eval(lines[7].split('fringe ')[1])
		barycorr       = json.loads(lines[14].split('barycorr')[1])
		print('fringe_param', fringe_param)
	else:
		barycorr       = json.loads(lines[13].split('barycorr')[1])
elif instrument == 'hires':
	custom_mask    = json.loads(lines[3].split('custom_mask')[1])
	priors         = ast.literal_eval(lines[4].split('priors ')[1])
	barycorr       = json.loads(lines[11].split('barycorr')[1])

# read the pre-determined fringe parameters
if include_fringe_model:
	A1=float(fringe_param['A1']) 
	A2=float(fringe_param['A2']) 
	A3=float(fringe_param['A3']) 
	Dos1=float(fringe_param['Dos1']) 
	Dos2=float(fringe_param['Dos2']) 
	Dos3=float(fringe_param['Dos3']) 
	R1=float(fringe_param['R1'])
	R2=float(fringe_param['R2'])
	R3=float(fringe_param['R3'])
	phi1=float(fringe_param['phi1'])
	phi2=float(fringe_param['phi2']) 
	phi3=float(fringe_param['phi3'])

	print(A1, Dos1, R1, phi1)
	print(A2, Dos2, R2, phi2)
	print(A3, Dos3, R3, phi3)


# no logg 5.5 for teff lower than 900
if 'btsettl08' in modelset.lower() and priors['teff_min'] < 900: logg_max = 5.0
else: logg_max = 5.5

## apply a custom mask
data.mask_custom(custom_mask=custom_mask)

# limit of the flux nuisance parameter: 5 percent of the median flux
A_const       = 0.05 * abs(np.median(data.flux))

# KPIC DRP has underestimated noise (no read/dark current noise)
if instrument == 'kpic':
	N_max = 20.0
else:
	N_max = 5.0

if 'btsettl08' in modelset.lower():
	limits         = { 
						'teff_min':max(priors['teff_min']-300,500), 'teff_max':min(priors['teff_max']+300,3500),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-A_const,							'A_max':A_const,
						'B_min':-0.6,                              	'B_max':0.6,
						'N_min':0.10,                               'N_max':N_max 				
					}

elif 'sonora' in modelset.lower():
	limits         = { 
						'teff_min':max(priors['teff_min']-300,200), 'teff_max':min(priors['teff_max']+300,2400),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-A_const,							'A_max':A_const,
						'B_min':-0.6,                              	'B_max':0.6,
						'N_min':0.10,                               'N_max':N_max 				
					}

elif 'phoenixaces' in modelset.lower():
	limits         = { 
						'teff_min':max(priors['teff_min']-300,2300), 'teff_max':min(priors['teff_max']+300,10000),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-A_const,							'A_max':A_const,
						'B_min':-0.6,								'B_max':0.6,
						'N_min':0.10,                               'N_max':N_max 				
					}

elif (modelset.upper() == 'PHOENIX-BTSETTL-CIFIST2011-2015') or (modelset.upper() == 'PHOENIX_BTSETTL_CIFIST2011_2015'):
	limits         = { 
						'teff_min':max(priors['teff_min']-300,2300), 'teff_max':min(priors['teff_max']+300,7000),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-A_const,							'A_max':A_const,
						'B_min':-0.6,								'B_max':0.6,
						'N_min':0.10,                               'N_max':N_max				
					}

# HIRES wavelength calibration is not that precise, release the constraint for the wavelength offset nuisance parameter
if data.instrument == 'hires':
	limits['B_min'] = -3.0 # Angstrom
	limits['B_max'] = +3.0 # Angstrom

if final_mcmc:
	limits['rv_min'] = priors['rv_min'] - 10
	limits['rv_max'] = priors['rv_max'] + 10

## add a pixel label for plotting
length1     = len(data.oriWave)
pixel       = np.delete(np.arange(length1), data.mask)
pixel       = pixel[pixel_start:pixel_end]

### mask the end pixels
data.wave     = data.wave[pixel_start:pixel_end]
data.flux     = data.flux[pixel_start:pixel_end]
data.noise    = data.noise[pixel_start:pixel_end]

if instrument.lower() != 'hires':
	tell_sp.wave  = tell_sp.wave[pixel_start:pixel_end]
	tell_sp.flux  = tell_sp.flux[pixel_start:pixel_end]
	tell_sp.noise = tell_sp.noise[pixel_start:pixel_end]

#if final_mcmc:
#	priors, limits         = mcmc_utils.generate_final_priors_and_limits(sp_type=sp_type, barycorr=barycorr, save_to_path1=save_to_path1)
#else:
#	priors, limits         = mcmc_utils.generate_initial_priors_and_limits(sp_type=sp_type)
#print(priors, limits)

if lsf is None:
	lsf           = smart.getLSF(tell_sp, alpha=alpha_tell, test=True, save_path=save_to_path)
#	print("LSF: ", lsf)
#else:
#	print("Use input lsf:", lsf)

# log file
log_path = save_to_path + '/mcmc_parameters.txt'
"""
file_log = open(log_path,"w+")
file_log.write("data_path {} \n".format(data.path))
file_log.write("tell_path {} \n".format(tell_sp.path))
file_log.write("data_name {} \n".format(data.name))
file_log.write("tell_name {} \n".format(tell_sp.name))
file_log.write("order {} \n".format(data.order))
file_log.write("custom_mask {} \n".format(custom_mask))
file_log.write("priors {} \n".format(priors))
file_log.write("ndim {} \n".format(ndim))
file_log.write("nwalkers {} \n".format(nwalkers))
file_log.write("step {} \n".format(step))
file_log.write("burn {} \n".format(burn))
file_log.write("pixel_start {} \n".format(pixel_start))
file_log.write("pixel_end {} \n".format(pixel_end))
file_log.write("barycorr {} \n".format(barycorr))
file_log.write("lsf {} \n".format(lsf))
file_log.write("med_snr {} \n".format(med_snr))
file_log.close()
"""

#########################################################################################
## for multiprocessing
#########################################################################################

print('include_fringe_model', include_fringe_model)

def lnlike(theta, data, lsf):
	"""
	Log-likelihood, computed from chi-squared.

	Parameters
	----------
	theta
	lsf
	data

	Returns
	-------
	-0.5 * chi-square + sum of the log of the noise

	"""

	## Parameters MCMC
	teff, logg, vsini, rv, am, pwv, A, B, N = theta #N noise prefactor
	#teff, logg, vsini, rv, , am, pwv, A, B, freq, amp, phase = theta

	if include_fringe_model:
		model = model_fit.makeModel(teff=teff, logg=logg, metal=0.0, vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=B, flux_offset=A,
			lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=am, pwv=pwv, include_fringe_model=include_fringe_model, instrument=instrument,
			A1=A1, A2=A2, A3=A3, Dos1=Dos1, Dos2=Dos2, Dos3=Dos3, R1=R1, R2=R2, R3=R3, phi1=phi1, phi2=phi2, phi3=phi3)
	else:
		model = model_fit.makeModel(teff=teff, logg=logg, metal=0.0, vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=B, flux_offset=A,
			lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=am, pwv=pwv, include_fringe_model=include_fringe_model, instrument=instrument)

	chisquare = smart.chisquare(data, model)/N**2

	return -0.5 * (chisquare + np.sum(np.log(2*np.pi*(data.noise*N)**2)))

def lnprior(theta, limits=limits):
	"""
	Specifies a flat prior
	"""
	## Parameters for theta
	teff, logg, vsini, rv, am, pwv, A, B, N = theta

	if  limits['teff_min']  < teff  < limits['teff_max'] \
	and limits['logg_min']  < logg  < limits['logg_max'] \
	and limits['vsini_min'] < vsini < limits['vsini_max']\
	and limits['rv_min']    < rv    < limits['rv_max']   \
	and limits['am_min']    < am    < limits['am_max']\
	and limits['pwv_min']   < pwv   < limits['pwv_max']\
	and limits['A_min']     < A     < limits['A_max']\
	and limits['B_min']     < B     < limits['B_max']\
	and limits['N_min']     < N     < limits['N_max']:
		return 0.0

	return -np.inf

def lnprob(theta, data, lsf):
		
	lnp = lnprior(theta)
		
	if not np.isfinite(lnp):
		return -np.inf
		
	return lnp + lnlike(theta, data, lsf)

pos = [np.array([	priors['teff_min']  + (priors['teff_max']   - priors['teff_min'] ) * np.random.uniform(), 
					priors['logg_min']  + (priors['logg_max']   - priors['logg_min'] ) * np.random.uniform(), 
					priors['vsini_min'] + (priors['vsini_max']  - priors['vsini_min']) * np.random.uniform(),
					priors['rv_min']    + (priors['rv_max']     - priors['rv_min']   ) * np.random.uniform(), 
					priors['am_min']    + (priors['am_max']     - priors['am_min'])    * np.random.uniform(),
					priors['pwv_min']   + (priors['pwv_max']    - priors['pwv_min'])   * np.random.uniform(),
					priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
					priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform(),
					priors['N_min']     + (priors['N_max']      - priors['N_min'])     * np.random.uniform()]) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, moves=emcee.moves.KDEMove())
time1 = time.time()
sampler.run_mcmc(pos, step, progress=True)
time2 = time.time()
## multiprocessing

set_start_method('fork')
with Pool() as pool:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, pool=pool)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, pool=pool,
			moves=emcee.moves.KDEMove())
	time1 = time.time()
	sampler.run_mcmc(pos, step, progress=True)
	time2 = time.time()

np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])

samples = sampler.chain[:, :, :].reshape((-1, ndim))

np.save(save_to_path + '/samples', samples)

print('total time: ',(time2-time1)/60,' min.')
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
print(sampler.acceptance_fraction)
autocorr_time = sampler.get_autocorr_time(discard=burn, quiet=True)
print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(autocorr_time)))
print(autocorr_time)

# create walker plots
sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
samples = np.load(save_to_path + '/samples.npy')

ylabels = ["$T_{\mathrm{eff}} (K)$","$\log{g}$(dex)","$v\sin{i}(km/s)$","$RV(km/s)$","$AM$", "pwv (mm)","$C_{F_{\lambda}}$ (cnt/s)","$C_{\lambda}$($\AA$)","$C_{noise}$"]


## create walker plots
plt.rc('font', family='sans-serif')
plt.tick_params(labelsize=30)
fig = plt.figure(tight_layout=True)
gs  = gridspec.GridSpec(ndim, 1)
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
if plot_show:
	plt.show()
plt.close()

# create array triangle plots
triangle_samples = sampler_chain[:, burn:, :].reshape((-1, ndim))
#print(triangle_samples.shape)

# create the final spectra comparison
teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, am_mcmc, pwv_mcmc, A_mcmc, B_mcmc, N_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
	zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

# add the summary to the txt file

# log file
log_path2 = save_to_path + '/mcmc_result.txt'

file_log2 = open(log_path2,"w+")
file_log2.write("teff_mcmc {}\n".format(str(teff_mcmc[0])))
file_log2.write("logg_mcmc {}\n".format(str(logg_mcmc[0])))
file_log2.write("vsini_mcmc {}\n".format(str(vsini_mcmc[0])))
file_log2.write("rv_mcmc {}\n".format(str(rv_mcmc[0]+barycorr)))
file_log2.write("am_mcmc {}\n".format(str(am_mcmc[0])))
file_log2.write("pwv_mcmc {}\n".format(str(pwv_mcmc[0])))
file_log2.write("A_mcmc {}\n".format(str(A_mcmc[0])))
file_log2.write("B_mcmc {}\n".format(str(B_mcmc[0])))
file_log2.write("N_mcmc {}\n".format(str(N_mcmc[0])))
file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
file_log2.write("am_mcmc_e {}\n".format(str(max(abs(am_mcmc[1]), abs(am_mcmc[2])))))
file_log2.write("pwv_mcmc_e {}\n".format(str(max(abs(pwv_mcmc[1]), abs(pwv_mcmc[2])))))
file_log2.write("A_mcmc_e {}\n".format(str(max(abs(A_mcmc[1]), abs(A_mcmc[2])))))
file_log2.write("B_mcmc_e {}\n".format(str(max(abs(B_mcmc[1]), abs(B_mcmc[2])))))
file_log2.write("N_mcmc_e {}\n".format(str(max(abs(N_mcmc[1]), abs(N_mcmc[2])))))
# upper and lower uncertainties
# upper uncertainties
file_log2.write("teff_mcmc_ue {}\n".format(str(abs(teff_mcmc[1]))))
file_log2.write("logg_mcmc_ue {}\n".format(str(abs(logg_mcmc[1]))))
file_log2.write("vsini_mcmc_ue {}\n".format(str(abs(vsini_mcmc[1]))))
file_log2.write("rv_mcmc_ue {}\n".format(str(abs(rv_mcmc[1]))))
file_log2.write("am_mcmc_ue {}\n".format(str(abs(am_mcmc[1]))))
file_log2.write("pwv_mcmc_ue {}\n".format(str(abs(pwv_mcmc[1]))))
file_log2.write("A_mcmc_ue {}\n".format(str(abs(A_mcmc[1]))))
file_log2.write("B_mcmc_ue {}\n".format(str(abs(B_mcmc[1]))))
file_log2.write("N_mcmc_ue {}\n".format(str(abs(N_mcmc[1]))))
# lower uncertainties
file_log2.write("teff_mcmc_le {}\n".format(str(abs(teff_mcmc[2]))))
file_log2.write("logg_mcmc_le {}\n".format(str(abs(logg_mcmc[2]))))
file_log2.write("vsini_mcmc_le {}\n".format(str(abs(vsini_mcmc[2]))))
file_log2.write("rv_mcmc_le {}\n".format(str(abs(rv_mcmc[2]))))
file_log2.write("am_mcmc_le {}\n".format(str(abs(am_mcmc[2]))))
file_log2.write("pwv_mcmc_le {}\n".format(str(abs(pwv_mcmc[2]))))
file_log2.write("A_mcmc_le {}\n".format(str(abs(A_mcmc[2]))))
file_log2.write("B_mcmc_le {}\n".format(str(abs(B_mcmc[2]))))
file_log2.write("N_mcmc_le {}\n".format(str(abs(N_mcmc[2]))))
file_log2.close()

#print(teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, am_mcmc, pwv_mcmc, A_mcmc, B_mcmc, N_mcmc)

triangle_samples[:,3] += barycorr

## triangular plots
plt.rc('font', family='sans-serif')
fig = corner.corner(triangle_samples, 
	labels=ylabels,
	truths=[teff_mcmc[0], 
	logg_mcmc[0],
	vsini_mcmc[0], 
	rv_mcmc[0]+barycorr, 
	am_mcmc[0],
	pwv_mcmc[0],
	A_mcmc[0],
	B_mcmc[0],
	N_mcmc[0]],
	quantiles=[0.16, 0.84],
	label_kwargs={"fontsize": 20})
plt.minorticks_on()
fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
if plot_show:
	plt.show()
plt.close()

teff  = teff_mcmc[0]
logg  = logg_mcmc[0]
z     = 0.0
vsini = vsini_mcmc[0]
rv    = rv_mcmc[0]
am    = am_mcmc[0]
pwv   = pwv_mcmc[0]
A     = A_mcmc[0]
B     = B_mcmc[0]
N     = N_mcmc[0]

if include_fringe_model:
	model, model_notell  = model_fit.makeModel(teff=teff, logg=logg, metal=0.0, 
		vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=B, flux_offset=A,
		lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=am, pwv=pwv, 
		include_fringe_model=include_fringe_model, instrument=instrument, output_stellar_model=True,
		A1=A1, A2=A2, A3=A3, Dos1=Dos1, Dos2=Dos2, Dos3=Dos3, R1=R1, R2=R2, R3=R3, phi1=phi1, phi2=phi2, phi3=phi3)
else:
	model, model_notell = model_fit.makeModel(teff=teff, logg=logg, metal=0.0, 
		vsini=vsini, rv=rv, tell_alpha=1.0, wave_offset=B, flux_offset=A,
		lsf=lsf, order=str(data.order), data=data, modelset=modelset, airmass=am, pwv=pwv, output_stellar_model=True, 
		include_fringe_model=include_fringe_model, instrument=instrument)

fig = plt.figure(figsize=(16,6))
ax1 = fig.add_subplot(111)
plt.rc('font', family='sans-serif')
plt.tick_params(labelsize=15)
ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8)
ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8)
ax1.plot(data.wave,data.flux,'k-',
	label='data',alpha=0.5)
ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8)
plt.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
plt.axhline(y=0,color='k',linestyle='-',linewidth=0.5)
plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
plt.ylabel("Flux ($cnts/s$)",fontsize=15)
plt.xlabel("$\lambda$ ($\AA$)",fontsize=15)
plt.figtext(0.89,0.85,str(data.header['OBJECT'])+' '+data.name+' O'+str(data.order),
	color='k',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=15)
plt.figtext(0.89,0.82,"$Teff \, {0}^{{+{1}}}_{{-{2}}}/ logg \, {3}^{{+{4}}}_{{-{5}}}/ en \, 0.0/ vsini \, {6}^{{+{7}}}_{{-{8}}}/ RV \, {9}^{{+{10}}}_{{-{11}}}$".format(\
	round(teff_mcmc[0]),
	round(teff_mcmc[1]),
	round(teff_mcmc[2]),
	round(logg_mcmc[0],1),
	round(logg_mcmc[1],3),
	round(logg_mcmc[2],3),
	round(vsini_mcmc[0],2),
	round(vsini_mcmc[1],2),
	round(vsini_mcmc[2],2),
	round(rv_mcmc[0]+barycorr,2),
	round(rv_mcmc[1],2),
	round(rv_mcmc[2],2)),
	color='C0',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
	round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
color='k',
horizontalalignment='right',
verticalalignment='center',
fontsize=12)
plt.minorticks_on()

ax2 = ax1.twiny()
ax2.plot(pixel, data.flux, color='w', alpha=0)
ax2.set_xlabel('Pixel',fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xlim(pixel[0], pixel[-1])
ax2.minorticks_on()
	
#plt.legend()
plt.savefig(save_to_path + '/spectrum.png', dpi=300, bbox_inches='tight')
if plot_show:
	plt.show()
plt.close()

# chi2 and dof in the log
log_path = save_to_path + '/mcmc_parameters.txt'
file_log = open(log_path,"a")
file_log.write("*** Below is the summary *** \n")
file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
file_log.write("mean_autocorrelation_time {0:.3f} \n".format(np.mean(autocorr_time)))
file_log.write("chi2 {} \n".format(round(smart.chisquare(data,model))))
file_log.write("dof {} \n".format(round(len(data.wave-ndim)/3)))
file_log.write("teff_mcmc {} K\n".format(str(teff_mcmc)))
file_log.write("logg_mcmc {} dex (cgs)\n".format(str(logg_mcmc)))
file_log.write("vsini_mcmc {} km/s\n".format(str(vsini_mcmc)))
file_log.write("rv_mcmc {} km/s\n".format(str(rv_mcmc)))
file_log.write("am_mcmc {}\n".format(str(am_mcmc)))
file_log.write("pwv_mcmc {}\n".format(str(pwv_mcmc)))
file_log.write("A_mcmc {}\n".format(str(A_mcmc)))
file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
file_log.write("N_mcmc {}\n".format(str(N_mcmc)))
file_log.close()


# excel summary file
#cat = pd.DataFrame(columns=['date_obs','date_name','tell_name','data_path','tell_path','save_path',
#							'model_date','model_time','data_mask','order','coadd','mjd','med_snr','lsf',
#							'barycorr','modelset','priors','limits','ndim','nwalkers','step','burn',
#							'rv','e_rv','ue_rv','le_rv','vsini','e_vsini','ue_vsini','le_vsini',
#							'teff','e_teff','ue_teff','le_teff','logg','e_logg','ue_logg','le_logg',
#							'am','e_am','ue_am','le_am','pwv','e_pwv','ue_pwv','le_pwv',
#							'cflux','e_cflux','ue_cflux','le_cflux','cwave','e_cwave','ue_cwave','le_cwave',
#							'cnoise','e_cnoise','ue_cnoise','le_cnoise','wave_cal_err','chi2','dof','acceptance_fraction','autocorr_time'])


med_snr      = np.nanmedian(data.flux/data.noise)
if instrument == 'nirspec':
	wave_cal_err = tell_sp.header['STD']
else:
	wave_cal_err = np.nan

cat = pd.DataFrame({'date_obs':date_obs,'date_name':sci_data_name,'tell_name':tell_data_name,
					'data_path':data_path,'tell_path':tell_path,'save_path':save_to_path,
					'model_date':today.isoformat(),'model_time':dt_string,'data_mask':str(custom_mask),
					'order':order,'coadd':coadd,'mjd':mjd,'med_snr':med_snr,'lsf':lsf, 'barycorr':barycorr,
					'modelset':modelset, 'priors':str(priors), 'limits':str(limits), 
					'ndim':ndim, 'nwalkers':nwalkers,'step':step, 'burn':burn,
					'rv':rv_mcmc[0]+barycorr, 'e_rv':max(rv_mcmc[1], rv_mcmc[2]), 'ue_rv':rv_mcmc[1], 'le_rv':rv_mcmc[2],
					'vsini':vsini_mcmc[0], 'e_vsini':max(vsini_mcmc[1], vsini_mcmc[2]), 'ue_vsini':vsini_mcmc[1], 'le_vsini':vsini_mcmc[2],
					'teff':teff_mcmc[0], 'e_teff':max(teff_mcmc[1],teff_mcmc[2]), 'ue_teff':teff_mcmc[1], 'le_teff':teff_mcmc[2],
					'logg':logg_mcmc[0], 'e_logg':max(logg_mcmc[1], logg_mcmc[2]), 'ue_logg':logg_mcmc[1], 'le_logg':logg_mcmc[2],
					'am':am_mcmc[0], 'e_am':max(am_mcmc[1], am_mcmc[2]), 'ue_am':am_mcmc[1], 'le_am':am_mcmc[2], 
					'pwv':pwv_mcmc[0], 'e_pwv':max(pwv_mcmc[1], pwv_mcmc[2]), 'ue_pwv':pwv_mcmc[1], 'le_pwv':pwv_mcmc[2],
					'cflux':A_mcmc[0], 'e_cflux':max(A_mcmc[1], A_mcmc[2]), 'ue_cflux':A_mcmc[1], 'le_cflux':A_mcmc[2],
					'cwave':B_mcmc[0], 'e_cwave':max(B_mcmc[1], B_mcmc[2]), 'ue_cwave':B_mcmc[1], 'le_cwave':B_mcmc[2], 
					'cnoise':N_mcmc[0],'e_cnoise':max(N_mcmc[1], N_mcmc[2]), 'ue_cnoise':N_mcmc[1], 'le_cnoise':N_mcmc[2], 
					'wave_cal_err':wave_cal_err, }, index=[0])

cat.to_excel(save_to_path + '/mcmc_summary.xlsx', index=False)

