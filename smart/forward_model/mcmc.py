import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
#from schwimmbad import MPIPool
#from multiprocessing import Pool
import smart
import corner
import os
import sys
import time
import copy
from astropy.io import fits
#os.environ["OMP_NUM_THREADS"] = "1"

def run_mcmc(sci_data, tell_data, priors, limits=None, ndim=7, nwalkers=50, step=500, burn=400, moves=2.0, pixel_start=10, pixel_end=-30, alpha_tell=1.0, modelset='btsettl08', save_to_path=None, plot_show=True, custom_mask=[], lsf=None):
	"""
	MCMC run for the science spectra. See the parameters in the makeModel function.

	Parameters
	----------

	sci_data  	: 	sepctrum object
					science data

	tell_data 	: 	spectrum object
					telluric data for calibrating the science spectra

	priors   	: 	dic
					keys are teff_min, teff_max, logg_min, logg_max, vsini_min, vsini_max, rv_min, rv_max, alpha_min, alpha_max,
					A_min, A_max, B_min, B_max

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

	Examples
	--------
	>>> import smart
	>>> order   = 33
	>>> data    = smart.Spectrum(name=sci_data_name, order=order, path=data_path)
	>>> tell_sp = smart.Spectrum(name=tell_data_name, order=data.order, path=tell_path)
	>>> priors  =  { 'teff_min':2400,  'teff_max':2800,
					 'logg_min':3.5,   'logg_max':5.5,
					 'vsini_min':0.0,  'vsini_max':100.0,
					 'rv_min':0.0,     'rv_max':200.0,
					 'alpha_min':0.9,  'alpha_max':1.1,
					 'A_min':-0.01,    'A_max':0.01,
					 'B_min':-0.01,    'B_max':0.01 		}
	>>> smart.run_mcmc(sci_data=data, tell_data=tell_sp, priors=priors)

	"""

	if save_to_path is not None:
		if not os.path.exists(save_to_path):
			os.makedirs(save_to_path)
	else:
		save_to_path = '.'

	if limits is None:
		limits = priors

	data          = copy.deepcopy(sci_data)
	tell_sp       = copy.deepcopy(tell_data)
	data.updateWaveSol(tell_sp)
	## apply a custom mask
	data.mask_custom(custom_mask=custom_mask)

	## add a pixel label for plotting
	length1     = len(data.oriWave)
	pixel       = np.delete(np.arange(length1),data.mask)
	pixel       = pixel[pixel_start:pixel_end]

	### mask the end pixels
	data.wave     = data.wave[pixel_start:pixel_end]
	data.flux     = data.flux[pixel_start:pixel_end]
	data.noise    = data.noise[pixel_start:pixel_end]

	tell_sp.wave  = tell_sp.wave[pixel_start:pixel_end]
	tell_sp.flux  = tell_sp.flux[pixel_start:pixel_end]
	tell_sp.noise = tell_sp.noise[pixel_start:pixel_end]

	# barycentric corrction
	barycorr      = smart.barycorr(data.header).value
	print("barycorr:",barycorr)

	if lsf is None:
		lsf           = smart.getLSF(tell_sp,alpha=alpha_tell, test=True, save_path=save_to_path)
		print("LSF: ", lsf)
	else:
		print("Use input lsf:", lsf)

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
	file_log.write("median_SNR {} \n".format(np.median(data.flux/data.noise)))
	file_log.close()

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
		teff, logg, vsini, rv, alpha, A, B = theta
		#teff, logg, vsini, rv, alpha, A, B, freq, amp, phase = theta

		model = smart.makeModel(teff, logg, 0.0, vsini, rv, alpha, B, A,
			lsf=lsf, order=data.order, data=data, modelset=modelset)

		chisquare = smart.chisquare(data, model)

		return -0.5 * (chisquare + np.sum(np.log(2*np.pi*data.noise**2)))

	def lnprior(theta, limits=limits):
		"""
		Specifies a flat prior
		"""
		## Parameters for theta
		teff, logg, vsini, rv, alpha, A, B = theta

		#limits =  { 'teff_min':2400,  'teff_max':2800,
		#			'logg_min':3.5,  'logg_max':5.5,
		#			'vsini_min':0.0,  'vsini_max':100.0,
		#			'rv_min':-200.0,    'rv_max':200.0,
		#			'alpha_min':0.0,  'alpha_max':2.0,
		#			'A_min':-1.0,     'A_max':1.0,
		#			'B_min':-0.6,     'B_max':0.6 		}
					#'lsf_min':4.5,    'lsf_max':6.5      }
					#'freq_min':0.0,   'freq_max':10.0,
					#'amp_min':0.0,    'amp_max':2.5,
					#'phase_min':-10.0,'phase_max':10.0 }

		if  limits['teff_min']  < teff  < limits['teff_max'] \
		and limits['logg_min']  < logg  < limits['logg_max'] \
		and limits['vsini_min'] < vsini < limits['vsini_max']\
		and limits['rv_min']    < rv    < limits['rv_max']   \
		and limits['alpha_min'] < alpha < limits['alpha_max']\
		and limits['A_min']     < A     < limits['A_max']\
		and limits['B_min']     < B     < limits['B_max']:
		#and limits['lsf_min']   < lsf   < limits['lsf_max']:
		#and limits['logg_min']  < logg  < limits['logg_max'] \
		#and limits['C_min']     < C     < limits['C_max']:
		#and priors['freq_min']  < freq  < priors['freq_max'] \
		#and priors['amp_min']   < amp   < priors['amp_max']  \
		#and priors['phase_min'] < phase < priors['phase_max']:
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
						priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform()]) for i in range(nwalkers)]
						#priors['lsf_min']   + (priors['lsf_max']    - priors['lsf_min'])   * np.random.uniform()]) for i in range(nwalkers)]
						#priors['freq_min']  + (priors['freq_max']   - priors['freq_min'] ) * np.random.uniform(),
						#priors['amp_min']   + (priors['amp_max']    - priors['amp_min']  ) * np.random.uniform(),
						#priors['phase_min'] + (priors['phase_max']  - priors['phase_min']) * np.random.uniform()]) for i in range(nwalkers)]
	
	#os.environ["OMP_NUM_THREADS"] = "1"
	#from multiprocessing import Pool
	#with Pool() as pool:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, pool=pool)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves)
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

	if ndim is 5:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$"]

	elif ndim is 6:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{wave}$"]

	elif ndim is 7:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{flux}$","$C_{wave}$"]

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
	teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
		zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

	# add the summary to the txt file
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
	file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
	#file_log.write("lsf_mcmc {}\n".format(str(lsf_mcmc)))
	#file_log.write("C_mcmc {}\n".format(str(C_mcmc)))
	#file_log.write("freq_mcmc {}\n".format(str(freq_mcmc)))
	#file_log.write("amp_mcmc {}\n".format(str(amp_mcmc)))
	#file_log.write("phase_mcmc {}\n".format(str(phase_mcmc)))
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
	file_log2.write("B_mcmc {}\n".format(str(B_mcmc[0])))
	file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
	file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
	file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
	file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
	file_log2.close()

	print(teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc)

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
		B_mcmc[0]],
		#lsf_mcmc[0]],
		#C_mcmc[0]],
		#freq_mcmc[0],
		#amp_mcmc[0],
		#phase_mcmc[0]],
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
	B     = B_mcmc[0]
	#lsf   = lsf_mcmc[0]
	#C     = C_mcmc[0]
	#freq  = freq_mcmc[0]
	#amp   = amp_mcmc[0]
	#phase = phase_mcmc[0]

	## new plotting model 
	## read in a model
	model        = smart.Model(teff=teff, logg=logg, feh=z, order=data.order, modelset=modelset)

	# apply vsini
	model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)    
	# apply rv (including the barycentric correction)
	model.wave   = smart.rvShift(model.wave, rv=rv)

	model_notell = copy.deepcopy(model)
	# apply telluric
	model        = smart.applyTelluric(model=model, alpha=alpha)
	# NIRSPEC LSF
	model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)

	# wavelength offset
	model.wave        += B
	
	# integral resampling
	model.flux   = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
	model.wave   = data.wave

	# contunuum correction
	model, cont_factor = smart.continuum(data=data, mdl=model, prop=True)

	# NIRSPEC LSF
	model_notell.flux  = smart.broaden(wave=model_notell.wave, flux=model_notell.flux, vbroad=lsf, rotate=False, gaussian=True)

	# wavelength offset
	model_notell.wave += B
	
	# integral resampling
	model_notell.flux  = np.array(smart.integralResample(xh=model_notell.wave, yh=model_notell.flux, xl=data.wave))
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
	ax1.plot(model_notell.wave,model_notell.flux, color='C1', linestyle='-', label='model no telluric',alpha=0.5)
	ax1.plot(data.wave,data.flux,'k-',
		label='data',alpha=0.5)
	ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8)
	plt.fill_between(data.wave,-data.noise,data.noise,facecolor='C0',alpha=0.5)
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

def run_mcmc2(sci_data, tell_data, priors, limits=None, ndim=8, nwalkers=50, step=600, burn=500, moves=2.0, pixel_start=10, pixel_end=-30, alpha_tell=1.0, modelset='btsettl08', save_to_path=None, plot_show=True, custom_mask=[], lsf=None):
	"""
	MCMC run for the science spectra. See the parameters in the makeModel function.

	Parameters
	----------

	sci_data  	: 	sepctrum object
					science data

	tell_data 	: 	spectrum object
					telluric data for calibrating the science spectra

	priors   	: 	dic
					keys are teff_min, teff_max, logg_min, logg_max, vsini_min, vsini_max, rv_min, rv_max, alpha_min, alpha_max,
					A_min, A_max, B_min, B_max

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

	Examples
	--------
	>>> import smart
	>>> order   = 33
	>>> data    = smart.Spectrum(name=sci_data_name, order=order, path=data_path)
	>>> tell_sp = smart.Spectrum(name=tell_data_name, order=data.order, path=tell_path)
	>>> priors  =  { 'teff_min':2400,  'teff_max':2800,
					 'logg_min':3.5,   'logg_max':5.5,
					 'vsini_min':0.0,  'vsini_max':100.0,
					 'rv_min':0.0,     'rv_max':200.0,
					 'alpha_min':0.9,  'alpha_max':1.1,
					 'A_min':-0.01,    'A_max':0.01,
					 'B_min':-0.01,    'B_max':0.01,
					 'lsf_min':4.8,    'lsf_max':4.81 		}
	>>> smart.run_mcmc(sci_data=data, tell_data=tell_sp, priors=priors)

	"""

	if save_to_path is not None:
		if not os.path.exists(save_to_path):
			os.makedirs(save_to_path)
	else:
		save_to_path = '.'

	if limits is None:
		limits = priors

	data          = copy.deepcopy(sci_data)
	tell_sp       = copy.deepcopy(tell_data)
	data.updateWaveSol(tell_sp)
	## apply a custom mask
	data.mask_custom(custom_mask=custom_mask)

	## add a pixel label for plotting
	length1     = len(data.oriWave)
	pixel       = np.delete(np.arange(length1),data.mask)
	pixel       = pixel[pixel_start:pixel_end]

	### mask the end pixels
	data.wave     = data.wave[pixel_start:pixel_end]
	data.flux     = data.flux[pixel_start:pixel_end]
	data.noise    = data.noise[pixel_start:pixel_end]

	tell_sp.wave  = tell_sp.wave[pixel_start:pixel_end]
	tell_sp.flux  = tell_sp.flux[pixel_start:pixel_end]
	tell_sp.noise = tell_sp.noise[pixel_start:pixel_end]

	# barycentric corrction
	barycorr      = smart.barycorr(data.header).value
	print("barycorr:",barycorr)

	if lsf is None:
		lsf           = smart.getLSF(tell_sp, alpha=alpha_tell, test=True, save_path=save_to_path)
		print("LSF: ", lsf)
	else:
		print("Use input lsf:", lsf)

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
	file_log.write("median_SNR {} \n".format(np.median(data.flux/data.noise)))
	file_log.close()

	def lnlike(theta, data):
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
		teff, logg, vsini, rv, alpha, A, B, lsf = theta
		#teff, logg, vsini, rv, alpha, A, B, freq, amp, phase = theta

		model = smart.makeModel(teff, logg, 0.0, vsini, rv, alpha, B, A,
			lsf=lsf, order=data.order, data=data, modelset=modelset)

		chisquare = smart.chisquare(data, model)

		return -0.5 * (chisquare + np.sum(np.log(2*np.pi*data.noise**2)))

	def lnprior(theta, limits=limits):
		"""
		Specifies a flat prior
		"""
		## Parameters for theta
		teff, logg, vsini, rv, alpha, A, B, lsf = theta

		#limits =  { 'teff_min':2400,  'teff_max':2800,
		#			'logg_min':3.5,  'logg_max':5.5,
		#			'vsini_min':0.0,  'vsini_max':100.0,
		#			'rv_min':-200.0,    'rv_max':200.0,
		#			'alpha_min':0.0,  'alpha_max':2.0,
		#			'A_min':-1.0,     'A_max':1.0,
		#			'B_min':-0.6,     'B_max':0.6 		}
					#'lsf_min':4.5,    'lsf_max':6.5      }
					#'freq_min':0.0,   'freq_max':10.0,
					#'amp_min':0.0,    'amp_max':2.5,
					#'phase_min':-10.0,'phase_max':10.0 }

		if  limits['teff_min']  < teff  < limits['teff_max'] \
		and limits['logg_min']  < logg  < limits['logg_max'] \
		and limits['vsini_min'] < vsini < limits['vsini_max']\
		and limits['rv_min']    < rv    < limits['rv_max']   \
		and limits['alpha_min'] < alpha < limits['alpha_max']\
		and limits['A_min']     < A     < limits['A_max']\
		and limits['B_min']     < B     < limits['B_max']\
		and limits['lsf_min']   < lsf   < limits['lsf_max']:
		#and limits['logg_min']  < logg  < limits['logg_max'] \
		#and limits['C_min']     < C     < limits['C_max']:
		#and priors['freq_min']  < freq  < priors['freq_max'] \
		#and priors['amp_min']   < amp   < priors['amp_max']  \
		#and priors['phase_min'] < phase < priors['phase_max']:
			return 0.0

		return -np.inf

	def lnprob(theta, data):
		
		lnp = lnprior(theta)
		
		if not np.isfinite(lnp):
			return -np.inf
		
		return lnp + lnlike(theta, data, )

	pos = [np.array([	priors['teff_min']  + (priors['teff_max']   - priors['teff_min'] ) * np.random.uniform(), 
						priors['logg_min']  + (priors['logg_max']   - priors['logg_min'] ) * np.random.uniform(), 
						priors['vsini_min'] + (priors['vsini_max']  - priors['vsini_min']) * np.random.uniform(),
						priors['rv_min']    + (priors['rv_max']     - priors['rv_min']   ) * np.random.uniform(), 
						priors['alpha_min'] + (priors['alpha_max']  - priors['alpha_min']) * np.random.uniform(),
						priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
						priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform(),
						priors['lsf_min']   + (priors['lsf_max']    - priors['lsf_min'])   * np.random.uniform()]) for i in range(nwalkers)]
						#priors['freq_min']  + (priors['freq_max']   - priors['freq_min'] ) * np.random.uniform(),
						#priors['amp_min']   + (priors['amp_max']    - priors['amp_min']  ) * np.random.uniform(),
						#priors['phase_min'] + (priors['phase_max']  - priors['phase_min']) * np.random.uniform()]) for i in range(nwalkers)]
	
	#os.environ["OMP_NUM_THREADS"] = "1"
	#from multiprocessing import Pool
	#with Pool() as pool:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, pool=pool)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, ), a=moves)
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

	if ndim is 5:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$"]

	elif ndim is 6:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{wave}$"]

	elif ndim is 7:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{flux}$","$C_{wave}$"]

	elif ndim is 8:
		ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{flux}$","$C_{wave}$","$LSF \, (km/s)$"]

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
	teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc, lsf_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
		zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

	# add the summary to the txt file
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
	file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
	file_log.write("lsf_mcmc {}\n".format(str(lsf_mcmc)))
	#file_log.write("lsf_mcmc {}\n".format(str(lsf_mcmc)))
	#file_log.write("C_mcmc {}\n".format(str(C_mcmc)))
	#file_log.write("freq_mcmc {}\n".format(str(freq_mcmc)))
	#file_log.write("amp_mcmc {}\n".format(str(amp_mcmc)))
	#file_log.write("phase_mcmc {}\n".format(str(phase_mcmc)))
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
	file_log2.write("B_mcmc {}\n".format(str(B_mcmc[0])))
	file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
	file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
	file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
	file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
	file_log2.write("lsf_mcmc {}\n".format(str(lsf_mcmc[0])))
	file_log2.write("lsf_mcmc_e {}\n".format(str(max(abs(lsf_mcmc[1]),abs(lsf_mcmc[2])))))
	file_log2.close()

	print(teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc, lsf_mcmc)

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
		B_mcmc[0],
		lsf_mcmc[0]],
		#C_mcmc[0]],
		#freq_mcmc[0],
		#amp_mcmc[0],
		#phase_mcmc[0]],
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
	B     = B_mcmc[0]
	lsf   = lsf_mcmc[0]
	#C     = C_mcmc[0]
	#freq  = freq_mcmc[0]
	#amp   = amp_mcmc[0]
	#phase = phase_mcmc[0]

	## new plotting model 
	## read in a model
	model        = smart.Model(teff=teff, logg=logg, feh=z, order=data.order, modelset=modelset)

	# apply vsini
	model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)    
	# apply rv (including the barycentric correction)
	model.wave   = smart.rvShift(model.wave, rv=rv)

	model_notell = copy.deepcopy(model)
	# apply telluric
	model        = smart.applyTelluric(model=model, alpha=alpha)
	# NIRSPEC LSF
	model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)

	# wavelength offset
	model.wave        += B
	
	# integral resampling
	model.flux   = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
	model.wave   = data.wave

	# contunuum correction
	model, cont_factor = smart.continuum(data=data, mdl=model, prop=True)

	# NIRSPEC LSF
	model_notell.flux  = smart.broaden(wave=model_notell.wave, flux=model_notell.flux, vbroad=lsf, rotate=False, gaussian=True)

	# wavelength offset
	model_notell.wave += B
	
	# integral resampling
	model_notell.flux  = np.array(smart.integralResample(xh=model_notell.wave, yh=model_notell.flux, xl=data.wave))
	model_notell.wave  = data.wave
	model_notell.flux *= cont_factor

	# flux offset
	model.flux        += A
	model_notell.flux += A

	# include fringe pattern
	#model.flux        *= (1 + amp * np.sin(freq * (model.wave - phase)))
	#model_notell.flux *= (1 + amp * np.sin(freq * (model.wave - phase)))

	plt.tick_params(labelsize=20)
	fig = plt.figure(figsize=(16,6))
	ax1 = fig.add_subplot(111)
	plt.rc('font', family='sans-serif')
	plt.tick_params(labelsize=15)
	ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8)
	ax1.plot(model_notell.wave,model_notell.flux, color='C1', linestyle='-', label='model no telluric',alpha=0.5)
	ax1.plot(data.wave,data.flux,'k-',
		label='data',alpha=0.5)
	ax1.plot(data.wave,data.flux-model.flux,'k-',alpha=0.8)
	plt.fill_between(data.wave,-data.noise,data.noise,facecolor='C0',alpha=0.5)
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


def run_mcmc3(sci_data, tell_data, priors, limits=None, ndim=8, nwalkers=50, step=500, burn=400, moves=2.0, pixel_start=10, pixel_end=-30, alpha_tell=1.0, modelset='btsettl08', save_to_path=None, plot_show=True, custom_mask=[], lsf=None):
	"""
	MCMC run for the science spectra. See the parameters in the makeModel function.

	Parameters
	----------

	sci_data  	: 	sepctrum object
					science data

	tell_data 	: 	spectrum object
					telluric data for calibrating the science spectra

	priors   	: 	dic
					keys are teff_min, teff_max, logg_min, logg_max, vsini_min, vsini_max, rv_min, rv_max, alpha_min, alpha_max,
					A_min, A_max, B_min, B_max

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

	Examples
	--------
	>>> import smart
	>>> order   = 33
	>>> data    = smart.Spectrum(name=sci_data_name, order=order, path=data_path)
	>>> tell_sp = smart.Spectrum(name=tell_data_name, order=data.order, path=tell_path)
	>>> priors  =  { 'teff_min':2400,  'teff_max':2800,
					 'logg_min':3.5,   'logg_max':5.5,
					 'vsini_min':0.0,  'vsini_max':100.0,
					 'rv_min':0.0,     'rv_max':200.0,
					 'alpha_min':0.9,  'alpha_max':1.1,
					 'A_min':-0.01,    'A_max':0.01,
					 'B_min':-0.01,    'B_max':0.01 		}
	>>> smart.run_mcmc(sci_data=data, tell_data=tell_sp, priors=priors)

	"""

	if save_to_path is not None:
		if not os.path.exists(save_to_path):
			os.makedirs(save_to_path)
	else:
		save_to_path = '.'

	if limits is None:
		limits = priors

	data          = copy.deepcopy(sci_data)
	tell_sp       = copy.deepcopy(tell_data)
	data.updateWaveSol(tell_sp)
	## apply a custom mask
	data.mask_custom(custom_mask=custom_mask)

	## add a pixel label for plotting
	length1     = len(data.oriWave)
	pixel       = np.delete(np.arange(length1),data.mask)
	pixel       = pixel[pixel_start:pixel_end]

	### mask the end pixels
	data.wave     = data.wave[pixel_start:pixel_end]
	data.flux     = data.flux[pixel_start:pixel_end]
	data.noise    = data.noise[pixel_start:pixel_end]

	tell_sp.wave  = tell_sp.wave[pixel_start:pixel_end]
	tell_sp.flux  = tell_sp.flux[pixel_start:pixel_end]
	tell_sp.noise = tell_sp.noise[pixel_start:pixel_end]

	# barycentric corrction
	barycorr      = smart.barycorr(data.header).value
	print("barycorr:",barycorr)

	if lsf is None:
		lsf           = smart.getLSF(tell_sp,alpha=alpha_tell, test=True, save_path=save_to_path)
		print("LSF: ", lsf)
	else:
		print("Use input lsf:", lsf)

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
	file_log.write("median_SNR {} \n".format(np.median(data.flux/data.noise)))
	file_log.close()

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
		teff, logg, vsini, rv, alpha, A, B, N = theta #N noise prefactor
		#teff, logg, vsini, rv, alpha, A, B, freq, amp, phase = theta

		model = smart.makeModel(teff, logg, 0.0, vsini, rv, alpha, B, A,
			lsf=lsf, order=data.order, data=data, modelset=modelset)

		chisquare = smart.chisquare(data, model)/N**2

		return -0.5 * (chisquare + np.sum(np.log(2*np.pi*(data.noise*N)**2)))

	def lnprior(theta, limits=limits):
		"""
		Specifies a flat prior
		"""
		## Parameters for theta
		teff, logg, vsini, rv, alpha, A, B, N = theta

		if  limits['teff_min']  < teff  < limits['teff_max'] \
		and limits['logg_min']  < logg  < limits['logg_max'] \
		and limits['vsini_min'] < vsini < limits['vsini_max']\
		and limits['rv_min']    < rv    < limits['rv_max']   \
		and limits['alpha_min'] < alpha < limits['alpha_max']\
		and limits['A_min']     < A     < limits['A_max']\
		and limits['B_min']     < B     < limits['B_max']\
		and limits['N_min']     < N     < limits['N_max']:
			return 0.0

		return -np.inf

	def lnprob(theta, data ,lsf):
		
		lnp = lnprior(theta)
		
		if not np.isfinite(lnp):
			return -np.inf
		
		return lnp + lnlike(theta, data, lsf)

	## multiprocessing
	#pool = MPIPool()
	#if not pool.is_master():
	#	pool.wait()
	#	sys.exit(0)

	pos = [np.array([	priors['teff_min']  + (priors['teff_max']   - priors['teff_min'] ) * np.random.uniform(), 
						priors['logg_min']  + (priors['logg_max']   - priors['logg_min'] ) * np.random.uniform(), 
						priors['vsini_min'] + (priors['vsini_max']  - priors['vsini_min']) * np.random.uniform(),
						priors['rv_min']    + (priors['rv_max']     - priors['rv_min']   ) * np.random.uniform(), 
						priors['alpha_min'] + (priors['alpha_max']  - priors['alpha_min']) * np.random.uniform(),
						priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
						priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform(),
						priors['N_min']     + (priors['N_max']      - priors['N_min'])     * np.random.uniform()]) for i in range(nwalkers)]

	#os.environ["OMP_NUM_THREADS"] = "1"
	from multiprocessing import Pool
	#with Pool() as pool:
	#	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves, pool=pool)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf), a=moves)#, threads=15)
	time1 = time.time()
	sampler.run_mcmc(pos, step, progress=True)
	time2 = time.time()

	#pool.close()

	print('total time: ',(time2-time1)/60,' min.')
	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
	print(sampler.acceptance_fraction)

	np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])

	samples = sampler.chain[:, :, :].reshape((-1, ndim))

	np.save(save_to_path + '/samples', samples)

	# create walker plots
	sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
	samples = np.load(save_to_path + '/samples.npy')

	ylabels = ["$Teff (K)$","$log \, g$","$vsin \, i \, (km/s)$","$rv \, (km/s)$","$alpha$","$C_{flux}$","$C_{wave}$","$C_{noise}$"]

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
	teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc, N_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
		zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

	# add the summary to the txt file
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
	file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
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
	file_log2.write("B_mcmc {}\n".format(str(B_mcmc[0])))
	file_log2.write("N_mcmc {}\n".format(str(N_mcmc[0])))
	file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
	file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
	file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
	file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
	file_log2.write("N_mcmc_e {}\n".format(str(max(abs(N_mcmc[1]), abs(N_mcmc[2])))))
	file_log2.close()

	print(teff_mcmc, logg_mcmc, vsini_mcmc, rv_mcmc, alpha_mcmc, A_mcmc, B_mcmc, N_mcmc)

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
	alpha = alpha_mcmc[0]
	A     = A_mcmc[0]
	B     = B_mcmc[0]
	N     = N_mcmc[0]

	## new plotting model 
	## read in a model
	model        = smart.Model(teff=teff, logg=logg, feh=z, order=data.order, modelset=modelset)

	# apply vsini
	model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)    
	# apply rv (including the barycentric correction)
	model.wave   = smart.rvShift(model.wave, rv=rv)

	model_notell = copy.deepcopy(model)
	# apply telluric
	model        = smart.applyTelluric(model=model, alpha=alpha)
	# NIRSPEC LSF
	model.flux   = smart.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)

	# wavelength offset
	model.wave        += B
	
	# integral resampling
	model.flux   = np.array(smart.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
	model.wave   = data.wave

	# contunuum correction
	model, cont_factor = smart.continuum(data=data, mdl=model, prop=True)

	# NIRSPEC LSF
	model_notell.flux  = smart.broaden(wave=model_notell.wave, flux=model_notell.flux, vbroad=lsf, rotate=False, gaussian=True)

	# wavelength offset
	model_notell.wave += B
	
	# integral resampling
	model_notell.flux  = np.array(smart.integralResample(xh=model_notell.wave, yh=model_notell.flux, xl=data.wave))
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
	ax1.plot(model_notell.wave,model_notell.flux, color='C1', linestyle='-', label='model no telluric',alpha=0.5)
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

def telluric_mcmc(tell_sp, nwalkers=30, step=400, burn=300, priors=None, moves=2.0, save=True, save_to_path=None, second_iter=False):
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

	Examples
	--------
	>>> import smart
	>>> tell_sp = smart.Spectrum(name='jan19s0024_calibrated',order=33)
	>>> smart.telluric_mcmc(tell_sp)

	"""

	## Initial parameters setup
	tell_data_name       = tell_sp.name
	tell_path            = tell_sp.path
	order                = tell_sp.order
	ndim                 = 4
	applymask            = False
	pixel_start          = 10
	pixel_end            = -30

	## add a pixel label for plotting
	length1     = len(tell_sp.oriWave)
	pixel       = np.delete(np.arange(length1),tell_sp.mask)
	pixel       = pixel[pixel_start:pixel_end]

	if priors is None:
		priors =  { 'lsf_min':4.0,  	'lsf_max':10.0,
					'alpha_min':0.3,  	'alpha_max':2.0,
					'A_min':-1.0,    	'A_max':1.0,
					'B_min':-0.5,		'B_max':0.5    	}

	if save_to_path is None:
		save_to_path = './mcmc'

	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)

	tell_sp.wave    = tell_sp.wave[pixel_start:pixel_end]
	tell_sp.flux    = tell_sp.flux[pixel_start:pixel_end]
	tell_sp.noise   = tell_sp.noise[pixel_start:pixel_end]

	data = copy.deepcopy(tell_sp)

	## MCMC functions
	def makeTelluricModel(lsf, alpha, flux_offset, wave_offset0, data=data):
		"""
		Make a telluric model as a function of LSF, alpha, and flux offset.
		"""
		data2               = copy.deepcopy(data)
		data2.wave          = data2.wave + wave_offset0
		#data2.wave          = data2.wave * (1 + wave_offset1) + wave_offset0
		telluric_model      = smart.convolveTelluric(lsf, data2, alpha=alpha)
		model               = smart.continuum(data=data2, mdl=telluric_model, deg=2)
		model.flux         += flux_offset

		return model

	## log file
	log_path = save_to_path + '/mcmc_parameters.txt'

	file_log = open(log_path,"w+")
	file_log.write("tell_path {} \n".format(tell_path))
	file_log.write("tell_name {} \n".format(tell_data_name))
	file_log.write("order {} \n".format(order))
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
		lsf, alpha, A, B = theta

		model = makeTelluricModel(lsf, alpha, A, B, data=data)

		chisquare = smart.chisquare(data, model)

		return -0.5 * (chisquare + np.sum(np.log(2*np.pi*data.noise**2)))

	def lnprior(theta):
		"""
		Specifies a flat prior
		"""
		## Parameters for theta
		lsf, alpha, A, B = theta

		limits =  { 'lsf_min':1.0  ,  'lsf_max':20.0,
					'alpha_min':0.3,  'alpha_max':2.0,
					'A_min':-100.0   ,  'A_max':100.0,
					'B_min':-0.5   ,  'B_max':0.5    }

		if  limits['lsf_min']   < lsf  < limits['lsf_max'] \
		and limits['alpha_min'] < alpha < limits['alpha_max']\
		and limits['A_min']     < A     < limits['A_max']\
		and limits['B_min']     < B     < limits['B_max']:
			return 0.0

		return -np.inf

	def lnprob(theta, data):
		lnp = lnprior(theta)

		if not np.isfinite(lnp):
			return -np.inf
		return lnp + lnlike(theta, data)

	pos = [np.array([priors['lsf_min']   + (priors['lsf_max']    - priors['lsf_min'] )  * np.random.uniform(), 
					 priors['alpha_min'] + (priors['alpha_max']  - priors['alpha_min']) * np.random.uniform(),
					 priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
					 priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform()]) for i in range(nwalkers)]

	#with Pool() as pool:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves, pool=pool)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves)
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

	ylabels = ["LSF (km/s)", "alpha", "flux offset", "wave offset0"]

	## create walker plots
	plt.rc('font', family='sans-serif')
	plt.tick_params(labelsize=30)
	fig = plt.figure(tight_layout=True)
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
	lsf_mcmc, alpha_mcmc, A_mcmc, B_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
		zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

	# add the summary to the txt file
	file_log = open(log_path,"a")
	file_log.write("*** Below is the summary *** \n")
	file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
	file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
	file_log.write("lsf_mcmc {} km/s\n".format(str(lsf_mcmc)))
	file_log.write("alpha_mcmc {}\n".format(str(alpha_mcmc)))
	file_log.write("A_mcmc {}\n".format(str(A_mcmc)))
	file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
	#file_log.write("C_mcmc {}\n".format(str(C_mcmc)))
	file_log.close()

	print(lsf_mcmc, alpha_mcmc, A_mcmc, B_mcmc)

	if '_' in tell_sp.name:
		tell_data_name = tell_sp.name.split('_')[0]

	## triangular plots
	plt.rc('font', family='sans-serif')
	fig = corner.corner(triangle_samples, 
		labels=ylabels,
		truths=[lsf_mcmc[0], 
		alpha_mcmc[0],
		A_mcmc[0],
		B_mcmc[0]],
		quantiles=[0.16, 0.84],
		label_kwargs={"fontsize": 20})
	plt.minorticks_on()
	fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
	#plt.show()
	plt.close()

	data2               = copy.deepcopy(data)
	data2.wave          = data2.wave + B_mcmc[0]
	telluric_model      = smart.convolveTelluric(lsf_mcmc[0], data, alpha=alpha_mcmc[0])
	model, pcont        = smart.continuum(data=data, mdl=telluric_model, deg=2, tell=True)
	model.flux         += A_mcmc[0]

	plt.tick_params(labelsize=20)
	fig = plt.figure(figsize=(20,8))
	ax1 = fig.add_subplot(111)
	ax1.plot(model.wave, model.flux, c='C3', ls='-', alpha=0.5)
	ax1.plot(model.wave, np.polyval(pcont, model.wave) + A_mcmc[0], c='C1', ls='-', alpha=0.5)
	ax1.plot(data.wave, data.flux, 'k-', alpha=0.5)
	ax1.plot(data.wave, data.flux-(model.flux+A_mcmc[0]),'k-', alpha=0.5)
	plt.figtext(0.89,0.86,"{} O{}".format(tell_data_name, order),
		color='k',
		horizontalalignment='right',
		verticalalignment='center',
		fontsize=12)
	plt.figtext(0.89,0.83,"${0}^{{+{1}}}_{{-{2}}}/{3}^{{+{4}}}_{{-{5}}}/{6}^{{+{7}}}_{{-{8}}}$".format(\
		round(lsf_mcmc[0],2),
		round(lsf_mcmc[1],2),
		round(lsf_mcmc[2],2),
		round(alpha_mcmc[0],2),
		round(alpha_mcmc[1],2),
		round(alpha_mcmc[2],2),
		round(A_mcmc[0],2),
		round(A_mcmc[1],2),
		round(A_mcmc[2],2)),
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

	ax2 = ax1.twiny()
	ax2.plot(pixel, data.flux, color='w', alpha=0)
	ax2.set_xlabel('Pixel',fontsize=20)
	ax2.tick_params(labelsize=15)
	ax2.set_xlim(pixel[0], pixel[-1])
	ax2.minorticks_on()
	
	plt.savefig(save_to_path+'/telluric_spectrum.png',dpi=300, bbox_inches='tight')
	#plt.show()
	plt.close()

	if save is True:
		data_path = tell_sp.path + '/' + tell_sp.name + '_' + str(tell_sp.order) + '_all.fits'
		with fits.open(data_path) as hdulist:
			hdulist[0].header['LSF']       = lsf_mcmc[0]
			hdulist[0].header['ALPHA']     = alpha_mcmc[0]
			#hdulist[0].header['WFIT0NEW'] += A_mcmc[0]
			try:
				hdulist.writeto(data_path, overwrite=True)
			except FileNotFoundError:
				hdulist.writeto(data_path)

	return lsf_mcmc[0]
#def run_mcmc_wrapper():

