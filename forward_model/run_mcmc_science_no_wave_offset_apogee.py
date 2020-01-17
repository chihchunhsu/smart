import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
import matplotlib.gridspec as gridspec
from astropy.io import fits
import emcee
#from schwimmbad import MPIPool
from multiprocessing import Pool
import nirspec_fmp as nsp
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
warnings.filterwarnings("ignore")

##############################################################################################
## This is the script to make the code multiprocessing, using arcparse to pass the arguments
## The code is run with 7 parameters, including Teff, logg, RV, vsini, telluric alpha, and 
## nuisance parameters for flux and noise (no wavelength parameters).
##############################################################################################

parser = argparse.ArgumentParser(description="Run the forward-modeling routine for science files of SDSS/APOGEE spectra",
	usage="run_mcmc_science.py sci_data_name tell_data_name data_path tell_path save_to_path lsf priors limits")

parser.add_argument("sci_data_name",metavar='sci',type=str,
    default=None, help="science data name", nargs="+")

parser.add_argument("data_path",type=str,
    default=None, help="science data path", nargs="+")

parser.add_argument("save_to_path",type=str,
    default=None, help="output path", nargs="+")

parser.add_argument("lsf",type=float,
    default=None, help="line spread function", nargs="+")

parser.add_argument("-outlier_rejection",metavar='--rej',type=float,
    default=3.0, help="outlier rejection based on the multiple of standard deviation of the residual; default 3.0")

parser.add_argument("-ndim",type=int,
    default=7, help="number of dimension; default 8")

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

parser.add_argument("-applymask",type=bool,
    default=False, help="apply a simple mask based on the STD of the average flux; default is False")

parser.add_argument("-plot_show",type=bool,
    default=False, help="show the MCMC plots; default is False")

parser.add_argument("-coadd",type=bool,
    default=False, help="coadd the spectra; default is False")

parser.add_argument("-coadd_sp_name",type=str,
    default=None, help="name of the coadded spectra")

parser.add_argument("-modelset",type=str,
    default='btsettl08', help="model set; default is btsettl08")

parser.add_argument("-instrument",type=str,
    default='apogee', help="instrument; default is btsettl08")

parser.add_argument("-final_mcmc", action='store_true', help="run final mcmc; default False")

args = parser.parse_args()

######################################################################################################

#date_obs               = str(args.date_obs[0])
sci_data_name          = str(args.sci_data_name[0])
data_path              = str(args.data_path[0])
save_to_path_base      = str(args.save_to_path[0])
lsf                    = float(args.lsf[0])
ndim, nwalkers, step   = int(args.ndim), int(args.nwalkers), int(args.step)
burn                   = int(args.burn)
moves                  = float(args.moves)
applymask              = args.applymask
pixel_start, pixel_end = int(args.pixel_start), int(args.pixel_end)
plot_show              = args.plot_show
outlier_rejection      = float(args.outlier_rejection)
modelset               = str(args.modelset)
instrument             = str(args.instrument)
final_mcmc             = args.final_mcmc

if final_mcmc:
	save_to_path1  = save_to_path_base + '/init_mcmc'
	save_to_path   = save_to_path_base + '/final_mcmc'

else:
	save_to_path   = save_to_path_base + '/init_mcmc'
	

#####################################

data        = nsp.Spectrum(name=sci_data_name, path=data_path, applymask=applymask, instrument=instrument)

sci_data    = data

"""
MCMC run for the science spectra. See the parameters in the makeModel function.

Parameters
----------

sci_data  	: 	sepctrum object
				science data

tell_data 	: 	spectrum object
				telluric data for calibrating the science spectra

priors   	: 	dic
				keys are teff_min, teff_max, logg_min, logg_max, vsini_min, vsini_max, rv_min, rv_max, alpha_min, alpha_max, A_min, A_max, B_min, B_max

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

# barycentric corrction
#barycorr      = nsp.barycorr(data.header).value
#print("barycorr:",barycorr)

## read the input custom mask and priors
lines          = open(save_to_path+'/mcmc_parameters.txt').read().splitlines()
custom_mask    = json.loads(lines[2].split('custom_mask')[1])
priors         = ast.literal_eval(lines[3].split('priors ')[1])


# no logg 5.5 for teff lower than 900
if priors['teff_min'] <= 1300: logg_max = 5.0
else: logg_max = 5.5

# limit of the flux nuisance parameter: 5 percent of the median flux
A_const       = 0.05 * abs(np.median(data.flux))

if modelset == 'btsettl08':
	limits         = { 
						'teff_min':max(priors['teff_min']-200,500), 'teff_max':min(priors['teff_max']+200,3500),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'alpha_min':0.1,                            'alpha_max':4.0,
						'A_min':-A_const,							'A_max':A_const,
						'N_min':0.10,                               'N_max':2.50 				
					}

elif modelset == 'phoenixaces':
	limits         = { 
						'teff_min':max(priors['teff_min']-200,2300), 'teff_max':min(priors['teff_max']+200,10000),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'alpha_min':0.1,                            'alpha_max':2.5,
						'A_min':-A_const,                           'A_max':A_const,
						'N_min':0.10,                               'N_max':2.50 				
					}

## apply a custom mask
data.mask_custom(custom_mask=custom_mask)

## add a pixel label for plotting
length1     = len(data.wave)
pixel       = np.delete(np.arange(length1),data.mask)
pixel       = pixel[pixel_start:pixel_end]

### mask the end pixels
data.wave     = data.wave[pixel_start:pixel_end]
data.flux     = data.flux[pixel_start:pixel_end]
data.noise    = data.noise[pixel_start:pixel_end]


#if final_mcmc:
#	priors, limits         = mcmc_utils.generate_final_priors_and_limits(sp_type=sp_type, barycorr=barycorr, save_to_path1=save_to_path1)
#else:
#	priors, limits         = mcmc_utils.generate_initial_priors_and_limits(sp_type=sp_type)
#print(priors, limits)

if lsf is None:
	lsf           = nsp.getLSF(tell_sp, alpha=alpha_tell, test=True, save_path=save_to_path)
#	print("LSF: ", lsf)
#else:
#	print("Use input lsf:", lsf)
"""
# log file
log_path = save_to_path + '/mcmc_parameters.txt'

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
file_log.close()
"""
#########################################################################################
## for multiprocessing
#########################################################################################
def makeModel(teff,logg,z,vsini,rv,alpha,wave_offset,flux_offset,**kwargs):
	"""
	Return a forward model.

	Parameters
	----------
	teff   : effective temperature
	
	data   : an input science data used for continuum correction

	Optional Parameters
	-------------------
	

	Returns
	-------
	model: a synthesized model
	"""

	# read in the parameters
	order      = kwargs.get('order', 33)
	modelset   = kwargs.get('modelset', 'btsettl08')
	instrument = kwargs.get('instrument', 'nirspec')
	lsf        = kwargs.get('lsf', 6.0)   # instrumental LSF
	tell       = kwargs.get('tell', True) # apply telluric
	data       = kwargs.get('data', None) # for continuum correction and resampling
	
	if data is not None and instrument == 'nirspec':
		order = data.order
		# read in a model
		#print('teff ',teff,'logg ',logg, 'z', z, 'order', order, 'modelset', modelset)
		#print('teff ',type(teff),'logg ',type(logg), 'z', type(z), 'order', type(order), 'modelset', type(modelset))
		model    = nsp.Model(teff=teff, logg=logg, feh=z, order=order, modelset=modelset, instrument=instrument)

	elif data is not None and instrument == 'apogee':
		model    = nsp.Model(teff=teff, logg=logg, feh=z, modelset=modelset, instrument=instrument)
	
	# wavelength offset
	#model.wave += wave_offset

	# apply vsini
	model.flux = nsp.broaden(wave=model.wave, 
		flux=model.flux, vbroad=vsini, rotate=True, gaussian=False)
	
	# apply rv (including the barycentric correction)
	model.wave = nsp.rvShift(model.wave, rv=rv)
	
	# apply telluric
	if tell is True:
		model = nsp.applyTelluric(model=model, alpha=alpha, airmass='1.5')
	# instrumental LSF
	model.flux = nsp.broaden(wave=model.wave, 
		flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)

	# add a fringe pattern to the model
	#model.flux *= (1+amp*np.sin(freq*(model.wave-phase)))

	# wavelength offset
	model.wave += wave_offset

	# integral resampling
	if data is not None:
		model.flux = np.array(nsp.integralResample(xh=model.wave, 
			yh=model.flux, xl=data.wave))
		model.wave = data.wave
		# contunuum correction
		model = nsp.continuum(data=data, mdl=model)

	# flux offset
	model.flux += flux_offset
	#model.flux **= (1 + flux_exponent_offset)

	return model
############################################################################

import numpy as np
import sys, os, os.path, time
from astropy.table import Table

FULL_PATH  = os.path.realpath(__file__)
BASE, NAME = os.path.split(FULL_PATH)
if modelset == 'btsettl08' and instrument == 'apogee':
	Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'

T1 = Table.read(Gridfile)
################################################################

def InterpModel(Teff, Logg, modelset='btsettl08', order=33, instrument='apogee'):

    FULL_PATH  = os.path.realpath(__file__)
    BASE, NAME = os.path.split(FULL_PATH)

    # Check the model set
    if instrument == 'nirspec':
        if modelset == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/NIRSPEC-O%s-RAW/'%order

        elif modelset == 'phoenixaces' :
            path = BASE + '/../libraries/phoenixaces/NIRSPEC-O%s-RAW/'%order

    elif instrument == 'apogee':
        if modelset == 'btsettl08':
            path = BASE + '/../libraries/btsettl08/APOGEE-RAW/'

    def bilinear_interpolation(x, y, points):
        '''Interpolate (x,y) from values associated with four points.

        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.

            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        points = sorted(points)               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return 10**((q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0))


    def GetModel(temp, logg, modelset='btsettl08', wave=False, instrument=instrument):
        feh, en = 0.00, 0.00
        if instrument == 'nirspec':
            if modelset == 'btsettl08':
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
            if modelset == 'phoenixaces':
                filename = 'phoenixaces_t{0:03d}'.format(int(temp.data[0])) + '_g{0:.2f}'.format(float(logg)) + '_z-{0:.2f}'.format(float(feh)) + '_en{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
        elif instrument == 'apogee':
            if modelset == 'btsettl08':
                filename = 'btsettl08_t'+ str(int(temp.data[0])) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_APOGEE-RAW.txt'

        Tab = Table.read(path+filename, format='ascii.tab', names=['wave', 'flux'])

        if wave:
            return Tab['wave']
        else:
            return Tab['flux']

    def myround(x, base=.5):
        return base * round(float(x)/base)

    def findlogg(logg):
        LoggArr = np.arange(2.5, 6, 0.5)
        dist = (LoggArr - logg)**2
        return LoggArr[np.argsort(dist)][0:2]

    if instrument == 'nirspec':
        if modelset == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams.csv'
        elif modelset == 'phoenixaces':
            Gridfile = BASE + '/../libraries/phoenixaces/phoenixaces_gridparams.csv'
    elif instrument == 'apogee':
        if modelset == 'btsettl08':
            Gridfile = BASE + '/../libraries/btsettl08/btsettl08_gridparams_apogee.csv'

    T1 = Table.read(Gridfile)

    # Check if the model already exists (grid point)
    if (Teff, Logg) in zip(T1['Temp'], T1['Logg']): 
        flux2  = GetModel(T1['Temp'][np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == Teff) & (T1['Logg'] == Logg))])
        waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == Teff) & (T1['Logg'] == Logg))], T1['Logg'][np.where((T1['Temp'] == Teff) & (T1['Logg'] == Logg))], wave=True)
        return waves2, flux2

    x1     = np.floor(Teff/100.)*100
    x2     = np.ceil(Teff/100.)*100
    y1, y2 = findlogg(Logg)

    # Get the nearest models to the gridpoint (Temp)
    x1 = T1['Temp'][np.where(T1['Temp'] <= x1)][-1]
    x2 = T1['Temp'][np.where(T1['Temp'] >= x2)][0]

    # Check if the gridpoint exists within the model ranges
    for x in [x1, x2]:
        for y in [y1, y2]:
            if (x, y) not in zip(T1['Temp'], T1['Logg']):
                print('No Model', x, y)
                return 1
    
    # Get the four points
    Points =  [ [np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))]), T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y2))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y2))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y2))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))]), T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y1))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y1))], modelset=modelset))],
                [np.log10(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))]), T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y2))], np.log10(GetModel(T1['Temp'][np.where( (T1['Temp'] == x2) & (T1['Logg'] == y2))], T1['Logg'][np.where((T1['Temp'] == x2) & (T1['Logg'] == y2))], modelset=modelset))],
              ]

    waves2 = GetModel(T1['Temp'][np.where( (T1['Temp'] == x1) & (T1['Logg'] == y1))], T1['Logg'][np.where((T1['Temp'] == x1) & (T1['Logg'] == y1))], wave=True, modelset=modelset)

    return waves2, bilinear_interpolation(np.log10(Teff), Logg, Points)



#########################################################################################
## for multiprocessing
#########################################################################################

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
	teff, logg, vsini, rv, alpha, A, N = theta #A: flux offset; N: noise prefactor

	## wavelength offset is set to 0
	model = makeModel(teff, logg, 0.0, vsini, rv, alpha, 0.0, A,
		lsf=lsf, data=data, modelset=modelset, instrument=instrument)
	print(len(data.wave),len(model.wave),len(data.noise),teff,logg,modelset,instrument)
	chisquare = nsp.chisquare(data, model)/N**2
	print(chisquare)
	return -0.5 * (chisquare + np.sum(np.log(2*np.pi*(data.noise*N)**2)))

def lnprior(theta, limits=limits):
	"""
	Specifies a flat prior
	"""
	## Parameters for theta
	teff, logg, vsini, rv, alpha, A, N = theta

	if  limits['teff_min']  < teff  < limits['teff_max'] \
	and limits['logg_min']  < logg  < limits['logg_max'] \
	and limits['vsini_min'] < vsini < limits['vsini_max']\
	and limits['rv_min']    < rv    < limits['rv_max']   \
	and limits['alpha_min'] < alpha < limits['alpha_max']\
	and limits['A_min']     < A     < limits['A_max']\
	and limits['N_min']     < N     < limits['N_max']:
		return 0.0

	return -np.inf

def lnprob(theta, data ,lsf):
		
	lnp = lnprior(theta)
		
	if not np.isfinite(lnp):
		return -np.inf
		
	return lnp + lnlike(theta, data, lsf)

pos = [np.array([	priors['teff_min']  + (priors['teff_max']   - priors['teff_min'] ) * np.random.uniform(), 
					priors['logg_min']  + (priors['logg_max']   - priors['logg_min'] ) * np.random.uniform(), 
					priors['vsini_min'] + (priors['vsini_max']  - priors['vsini_min']) * np.random.uniform(),
					priors['rv_min']    + (priors['rv_max']     - priors['rv_min']   ) * np.random.uniform(), 
					priors['alpha_min'] + (priors['alpha_max']  - priors['alpha_min']) * np.random.uniform(),
					priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
					priors['N_min']     + (priors['N_max']      - priors['N_min'])     * np.random.uniform()]) for i in range(nwalkers)]

print(priors)
print(pos)

## multiprocessing

with Pool() as pool:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, pool=pool)
	time1 = time.time()
	sampler.run_mcmc(pos, step, progress=True)
	time2 = time.time()

print('total time: ',(time2-time1)/60,' min.')
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
print(sampler.acceptance_fraction)

np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])

samples = sampler.chain[:, :, :].reshape((-1, ndim))

np.save(save_to_path + '/samples', samples)

# create walker plots
sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
samples = np.load(save_to_path + '/samples.npy')

ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{flux}$","$C_{noise}$"]

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
teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, N_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
	zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

# add the summary to the txt file
log_path = save_to_path + '/mcmc_parameters.txt'
file_log = open(log_path,"a")
file_log.write("*** Below is the summary *** \n")
file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
file_log.write("teff_mcmc {} K\n".format(str(teff_mcmc)))
file_log.write("logg_mcmc {} dex (cgs)\n".format(str(logg_mcmc)))
file_log.write("vsini_mcmc {} km/s\n".format(str(vsini_mcmc)))
file_log.write("rv_mcmc {} km/s\n".format(str(rv_mcmc)))
file_log.write("alpha_mcmc {}\n".format(str(alpha_mcmc)))
file_log.write("A_mcmc {}\n".format(str(A_mcmc)))
file_log.write("N_mcmc {}\n".format(str(N_mcmc)))
file_log.close()

# log file
log_path2 = save_to_path + '/mcmc_result.txt'

file_log2 = open(log_path2,"w+")
file_log2.write("teff_mcmc {}\n".format(str(teff_mcmc[0])))
file_log2.write("logg_mcmc {}\n".format(str(logg_mcmc[0])))
file_log2.write("vsini_mcmc {}\n".format(str(vsini_mcmc[0])))
file_log2.write("rv_mcmc {}\n".format(str(rv_mcmc[0]+barycorr)))
file_log2.write("alpha_mcmc {}\n".format(str(alpha_mcmc[0])))
file_log2.write("A_mcmc {}\n".format(str(A_mcmc[0])))
file_log2.write("N_mcmc {}\n".format(str(N_mcmc[0])))
file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
file_log2.write("N_mcmc_e {}\n".format(str(max(abs(N_mcmc[1]), abs(N_mcmc[2])))))
file_log2.close()

#print(teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc, N_mcmc)

triangle_samples[:,3] += barycorr

## triangular plots
plt.rc('font', family='sans-serif')
fig = corner.corner(triangle_samples, 
	labels=ylabels,
	truths=[teff_mcmc[0], 
	logg_mcmc[0],
	vsini_mcmc[0], 
	rv_mcmc[0]+barycorr, 
	alpha_mcmc[0],
	A_mcmc[0],
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
alpha = alpha_mcmc[0]
A     = A_mcmc[0]
N     = N_mcmc[0]

## new plotting model 
## read in a model
model        = nsp.Model(teff=teff, logg=logg, feh=z, order=data.order, modelset=modelset, instrument=instrument)

# apply vsini
model.flux   = nsp.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)    
# apply rv (including the barycentric correction)
model.wave   = nsp.rvShift(model.wave, rv=rv)

model_notell = copy.deepcopy(model)
# apply telluric
model        = nsp.applyTelluric(model=model, alpha=alpha)
# NIRSPEC LSF
model.flux   = nsp.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)
	
# integral resampling
model.flux   = np.array(nsp.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
model.wave   = data.wave

# contunuum correction
model, cont_factor = nsp.continuum(data=data, mdl=model, prop=True)

# NIRSPEC LSF
model_notell.flux  = nsp.broaden(wave=model_notell.wave, flux=model_notell.flux, vbroad=lsf, rotate=False, gaussian=True)
	
# integral resampling
model_notell.flux  = np.array(nsp.integralResample(xh=model_notell.wave, yh=model_notell.flux, xl=data.wave))
model_notell.wave  = data.wave
model_notell.flux *= cont_factor

# flux offset
model.flux        += A
model_notell.flux += A

# include fringe pattern
#model.flux        *= (1 + amp * np.sin(freq * (model.wave - phase)))
#model_notell.flux *= (1 + amp * np.sin(freq * (model.wave - phase)))

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
	round(nsp.chisquare(data,model)), round(len(data.wave-ndim)/3)),
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
