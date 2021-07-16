import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import os
from astropy.io import fits
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.special import wofz
import time
import sys
import smart
from .cal_param import cal_param_nirspec

FULL_PATH  = os.path.realpath(__file__)
BASE = os.path.split(os.path.split(os.path.split(FULL_PATH)[0])[0])[0]

# Set plotting parameters
plt.rc('font', family='sans-serif')


def waveSolution(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4, **kwargs):
    """
    Calculate the wavelength solution givene parameters of a 
    fourth order polynomial.
	The equation is modified from NSDRP_Software_Design p.35.

	Parameters
	----------
	wfit0-5: float
			 The fitting parameters in NSDRP_Software_Design p.35
	c3     : float
	         The coefficient of the cubic term of the polynomial
	c4     : float
			 The coefficient of the fourth-power term of the polynomial
	pixel  : int/array
			 The pixel number as the input of wavelength solution
    order  : int
             The order number as the input of wavelength

    Returns
    -------
    The wavelength solution: array-like


    """
    order = kwargs.get('order', None)

    wave_sol = wfit0 + wfit1*pixel + wfit2*pixel**2 + \
               (wfit3 + wfit4*pixel + wfit5*pixel**2)/order + \
               c3*pixel**3 + c4*pixel**4

    return wave_sol



def getTelluric(wavelow, wavehigh, **kwargs):
	"""
	Get a telluric spectrum.

	Parameters
	----------
	wavelow:  int
			  lower bound of the wavelength range

	wavehigh: int
	          upper bound of the wavelength range

	Optional Parameters
	-------------------
	airmass:  str
			  airmass of the telluric model, either 1.0 or 1.5
	alpha:    float
			  the power alpha parameter of the telluric model

	Returns
	-------
	telluric: model object
			  a telluric model with wavelength and flux


	Examples
	--------
	>>> import smart
	>>> telluric = smart.getTelluric(wavelow=22900,wavehigh=23250)

	"""
	
	airmass = kwargs.get('airmass', '1.5')
	alpha   = kwargs.get('alpha', 1)
	method  = kwargs.get('method', 'pwv')
	pwv     = kwargs.get('pwv', '1.5')
	am_key  = {'1.0':'10','1.5':'15','2.0':'20','2.5':'25','3.0':'30'}
	pwv_key = {'0.5':'005','1.0':'010','1.5':'015','2.5':'025',
	           '3.5':'035','5.0':'050','7.5':'075','10.0':'100','20.0':'200'}
	if method == 'pwv':
		tfile = 'pwv_R300k_airmass{}/LBL_A{}_s0_w{}_R0300000_T.fits'.format(airmass, 
			am_key[airmass],pwv_key[pwv])
	tellurics = fits.open(BASE + '/smart/libraries/telluric/'\
	 + tfile)

	telluric      = smart.Model()
	telluric.wave = np.array(tellurics[1].data['lam'] * 10000)
	telluric.flux = np.array(tellurics[1].data['trans'])**(alpha)

	# select the wavelength range
	criteria      = (telluric.wave > wavelow) & (telluric.wave < wavehigh)
	telluric.wave = telluric.wave[criteria]
	telluric.flux = telluric.flux[criteria]
	return telluric



def xcorrTelluric(data, model, shift, start_pixel, width, lsf):
	"""
	Calculate the cross-correlation of the telluric model and data.
	
	The shift (in Angstrom) acts on the model, and this function calculates given
	a starting pixel number and a window width (in pixels).

	This function skips the step of LSF convolution to speed up the calculation.
	
	Parameters
	----------
	data 	   	: 	spectrum object 
		        	telluric data
	model      	:	model object 
		   		 	telluric model
	shift      	: 	float 
		   		 	wavelength shift of the telluric MODEL in (ANGSTROM)
	start_pixel : 	int
				 	starting pixel number to compute the xcorr
	width       : 	int
				 	window width to compute the xcorr
	lsf 		: 	float
				 	the line-spread function for the instrument

	Returns
	-------
	xcorr 		: 	float
				  	cross-corelation value
	"""
	# shift the wavelength on the telluric model
	# minus sign means we want to know the shift of the data
	model2      = copy.deepcopy(model)
	model2.wave = model.wave - shift  # This is where we apply the X-corr shift
	#print('start_pixel, width: ', start_pixel, width)
	# select a range of wavelength to compute x-correlation value
	# the model has the range within that window
	model_low   = data.wave[start_pixel]       - 100
	model_high  = data.wave[start_pixel+width] + 100
	condition   = np.where( (model2.wave < model_high) & (model2.wave > model_low) )
	model2.wave = model2.wave[condition]
	model2.flux = model2.flux[condition]

	## LSF of the intrument
	model2.flux = smart.broaden(wave=model2.wave, flux=model2.flux, vbroad=lsf, 
		                      rotate=False, gaussian=True)

	# resampling the telluric model 
	# Note that +1 means the total xcorr values should be computed as -width/2 + center + width/2
	model2.flux = np.array(smart.integralResample(xh=model2.wave, yh=model2.flux, 
		                                        xl=data.wave[start_pixel:start_pixel+width+1]))
	model2.wave = data.wave[start_pixel:start_pixel+width+1]

	d = data.flux[start_pixel:start_pixel+width+1]
	##the model is selected in the pixel range in the beginning
	#m = model2.flux[start_pixel:start_pixel+width]
	m = model2.flux
	xcorr = np.inner(d, m)/(np.average(d)*np.average(m))

	return xcorr



def pixelWaveShift(data, model, start_pixel, window_width=40, delta_wave_range=20,
	               model2=None, test=False, testname=None, counter=None, **kwargs):
	"""
	Find the max cross-correlation and compute the pixel to wavelength shift.
	
	Parameters
	----------
	data
	model:  MUST BEFORE resampling and LSF broadening
	model2: model AFTER resampling and LSF broadening (to increase computation speed)


	Returns
	-------
	best wavelength shift: 	float
					  		the wavelength correction after xcorr

	"""
	# the step for wavelength shift, default=0.1 Angstrom
	step              = kwargs.get('step', 0.1)
	lsf               = kwargs.get('lsf')
	pixel_range_start = kwargs.get('pixel_range_start', 0)
	pixel_range_end   = kwargs.get('pixel_range_end', -1)
	length1           = kwargs.get('length1', len(data.oriWave))
	pixel             = kwargs.get('pixel', np.delete(np.arange(length1), data.mask))
	#pixel             = np.delete(np.arange(length1), data.mask)
	#pixel             = pixel[pixel_range_start:pixel_range_end]

	xcorr_list        = [] # save the xcorr values

	if model2 is None:
		model2 = model
	#if start_pixel < 400: delta_wave_range = 2
	# select the range of the pixel shift to compute the max xcorr
	for i in np.arange(-delta_wave_range, delta_wave_range+step, step):
		# propagate the best pixel shift
		if len(data.bestshift) > 1:
			j = data.bestshift[counter] + i
		else:
			j = i	
		xcorr = smart.xcorrTelluric(data, model, j, start_pixel, window_width, lsf)

		#print("delta wavelength shift:{}, xcorr value:{}".format(i,xcorr))
		xcorr_list.append(xcorr)

	#print("xcorr list:",xcorr_list)
	best_shift    = np.arange(-delta_wave_range, delta_wave_range+step, step)[np.argmax(xcorr_list)]
	central_pixel = start_pixel + window_width//2

	# parameters setup for plotting
	#plt.rc('text', usetex=True)
	#plt.rc('font', family='sans-serif')
	linewidth = 0.5

	if test:
		fig = plt.figure(figsize=(12,8))
		gs1 = gridspec.GridSpec(4, 4)
		ax1 = plt.subplot(gs1[0:2, :])
		ax2 = plt.subplot(gs1[2:, 0:2])
		ax3 = plt.subplot(gs1[2:, 2:])

		#print(data.wave, len(data.wave))
		#print(data.flux, len(data.flux))
		
		ax1.plot(data.wave, data.flux, color='black', linestyle='-', 
			     label='telluric data', alpha=0.5, linewidth=linewidth)
		ax1.plot(model2.wave, model2.flux, 'r-' , label='telluric model',
			     alpha=0.5, linewidth=linewidth)
		ax1.set_xlabel("Wavelength ($\AA$)")
		ax1.set_ylabel('Transmission')
		ax1.set_ylim(0,1.1)
		ax1.set_xlim(data.wave[0]-10, data.wave[-1]+10)
		ax1.set_title("Cross-Correlation Plot, pixels start at {} with width {}"\
			          .format(start_pixel, window_width))
		#ax1.set_title('Telluric Spectra Region for Cross-Correlation')
		ax1.axvline(x=data.wave[start_pixel], linestyle='--', color='blue',
			        linewidth=linewidth)
		ax1.axvline(x=data.wave[start_pixel+window_width], linestyle='--',
			        color='blue', linewidth=linewidth)
		ax1.get_xaxis().get_major_formatter().set_scientific(False)
		ax1.legend()
		ax1t = ax1.twiny()
		ax1t.set_xlim(data.wave[0], data.wave[-1])
		try:
			ax1t.set_xticks([data.wave[np.where(pixel==200)],
							 data.wave[np.where(pixel==400)],
							 data.wave[np.where(pixel==600)],
							 data.wave[np.where(pixel==800)],
							 data.wave[np.where(pixel==1000)]])
			ax1t.set_xticklabels(['200','400','600','800','1000'])
			ax1t.set_xlabel("Pixel")
		except ValueError:
			# pass if some pixels are masked
			pass
		except IndexError:
			pass

		ax2.plot(data.wave, data.flux, color='black', linestyle='-', 
				 label='telluric data', alpha=0.5, linewidth=linewidth)
		ax2.plot(model2.wave, model2.flux, 'r-' , label='telluric model',
				 alpha=0.5, linewidth=linewidth)
		ax2.set_xlabel("Wavelength ($\AA$)")
		ax2.set_ylabel('Transmission')
		ax2.set_ylim(0,1.1)
		ax2.set_xlim(data.wave[start_pixel]-0.1,
					 data.wave[start_pixel+window_width]+0.1)
		ax2.axvline(x=data.wave[start_pixel], linestyle='--',
					color='blue', linewidth=linewidth)
		ax2.axvline(x=data.wave[start_pixel+window_width], linestyle='--',
					color='blue', linewidth=linewidth)
		ax2.get_xaxis().get_major_formatter().set_scientific(False)
		ax2.legend()
		ax2t = ax2.twiny()
		ax2t.set_xlim(data.wave[start_pixel]-0.1,
					  data.wave[start_pixel+window_width]+0.1)
		ax2t.set_xticks([data.wave[start_pixel],
					  	 data.wave[start_pixel+window_width]])
		ax2t.set_xticklabels([str(start_pixel),
							  str(start_pixel+window_width)])
		ax2t.set_xlabel("Pixel")

	# pass if the max shift is equal to the delta wavelength shift
	# in this case the gaussian fit is meaningless
	if np.absolute(np.argmax(xcorr_list)) is delta_wave_range/step:
		pass
	else:
		x = np.arange(-delta_wave_range, delta_wave_range+step, step)
		y = xcorr_list
		# interpolate the xcorr and find the local minimum near the 
		# best shift
		xcorr_int_y = UnivariateSpline(x, xcorr_list, k=4, s=0)
		xcorr_int_x = np.arange(x[0], x[-1], 1000)

		# select the range of elements for the gaussian fitting
		# percent for the guassian fitting
		xcorr_fit_percent = 0.8
		condition         = y>np.min(y)+(np.max(y)-np.min(y))*xcorr_fit_percent
		x_select          = np.select([condition],[x])[condition]
		y_select          = np.select([condition],[y])[condition]
		if test is True:
			print("start pixel: {}".format(start_pixel))
		#print("x_select:",x_select)
		#print("y_select:",y_select)
		# select the values only around central central maximum
		#roots = xcorr_int_y.derivative().roots()
		#if len(roots) is 0:
		#	pass
		#elif len(roots) is 1:
		#	if roots[0] < best_shift:
		#		condition1 = x_select>roots[0]
		#		x_select = np.select([condition1],[x_select])[condition1]
		#		y_select = np.select([condition1],[y_select])[condition1]
		#	elif roots[0] > best_shift:
		#		condition1 = x_select<roots[0]
		#		x_select = np.select([condition1],[x_select])[condition1]
		#		y_select = np.select([condition1],[y_select])[condition1]
		#else:
		#	root_index = np.searchsorted(roots, best_shift, side='left')
		#	if root_index is 0:
		#		if roots[0] > best_shift:
		#			condition1 = x_select<roots[0]
		#			x_select = np.select([condition1],[x_select])[condition1]
		#			y_select = np.select([condition1],[y_select])[condition1]
		#		elif roots[0] < best_shift:
		#			condition1 = x_select>roots[0]
		#			x_select = np.select([condition1],[x_select])[condition1]
		#			y_select = np.select([condition1],[y_select])[condition1]
		#	else:
		#		root_right = roots[root_index]
		#		root_left  = roots[root_index-1]
		#		condition1 = x_select<root_right
		#		x_select = np.select([condition1],[x_select])[condition1]
		#		y_select = np.select([condition1],[y_select])[condition1]
		#		condition2 = x_select>root_left
		#		x_select = np.select([condition2],[x_select])[condition2]
		#		y_select = np.select([condition2],[y_select])[condition2]
		#	print("len(y_select):",len(y_select))

		diff = np.diff(x_select)
		b    = np.where(diff>1)[0]
		if len(b) is 0:
			pass
		elif len(b) is 1:
			k = b[0]
			l = x_select[k]
			if l < best_shift:
				condition1 = x_select>l
				x_select   = np.select([condition1],[x_select])[condition1]
				y_select   = np.select([condition1],[y_select])[condition1]
			elif x_select[k] > best_shift:
				condition1 = x_select<l
				x_select   = np.select([condition1],[x_select])[condition1]
				y_select   = np.select([condition1],[y_select])[condition1]
		else:
			for k in b:
				l_list = []
				l_list.append(x_select[k])
			for l in l_list:	
				if l < best_shift:
					condition1 = x_select>l
					x_select   = np.select([condition1],[x_select])[condition1]
					y_select   = np.select([condition1],[y_select])[condition1]
				elif l > best_shift:
					condition1 = x_select<l
					x_select   = np.select([condition1],[x_select])[condition1]
					y_select   = np.select([condition1],[y_select])[condition1]
			#print("x_select after:",x_select)
			#print("y_select after:",y_select)

		n      = len(xcorr_list)        #the number of data
		mean0  = best_shift             #note this correction
		sigma0 = sum((x-mean0)**2)/n    #note this correction

		# initial parameters of selected xcorr for the gaussian fit
		n2     = len(y_select)
		mean2  = best_shift
		try:
			sigma2 = sum((x_select-mean2)**2)/n2
		except ZeroDivisionError:
			pass

		def gaus(x,a,x0,sigma):
			return a*np.e**(-(x-x0)**2. / (2.*sigma**2))

		try:
			popt, pcov = curve_fit(gaus, x, y, p0=[np.max(xcorr_list), mean0, sigma0])
			if np.absolute(popt[1]) >= delta_wave_range:
				popt[1]  = best_shift
			#popt2,pcov2 = curve_fit(gaus,x_select,y_select,p0=[np.max(y_select),mean2,sigma2])
			popt2, pcov2 = curve_fit(gaus, x_select, y_select)
			if np.absolute(popt2[1]) >= delta_wave_range:
				popt2[1] = best_shift

			if test:
				#ax3.plot(x, gaus(x,popt[0],popt[1],popt[2]),'c-',label='gaussian fit')
				#ax3.plot([popt[1],popt[1]],[float(np.min(xcorr_list)),float(np.max(xcorr_list))],'c--',label="gaussian fitted pixel:{}".format(popt[1]))
				ax3.plot(x_select, y_select, color='fuchsia',
					     label="{} percent range".format(xcorr_fit_percent*100))
			
				ax3.plot(x_select,gaus(x_select, popt2[0], popt2[1], popt2[2]), color='olive',
					     label='gaussian fit for selected parameters, shift:{}'.format(popt2[1]),
					     alpha=0.5)
				ax3.axvline(x=popt2[1], linewidth=0.5, linestyle='--', color='olive')
				#for root in xcorr_int_y.derivative().roots():
				#	ax3.axvline(x=root, linewidth=0.3, color='purple')
			# replace the fitted gaussian value

			#the threshold for replacing the pixel shift with gaussian fit
			replace_shift_criteria = 0.5 
			if np.absolute(best_shift-popt2[1]) < replace_shift_criteria:
				best_shift = popt2[1]
			else:
				try:
					condition3 = x_select < np.argmax(x_select)+3
					x_select2  = np.select([condition3],[x_select])[condition3]
					y_select2  = np.select([condition3],[y_select])[condition3]
					condition4 = x_select > np.argmax(x_select2)-3
					x_select2  = np.select([condition4],[x_select2])[condition4]
					y_select2  = np.select([condition4],[y_select2])[condition4]
				
				except ValueError:
					try:
						condition3 = x_select < np.argmax(x_select)+2
						x_select2  = np.select([condition3],[x_select])[condition3]
						y_select2  = np.select([condition3],[y_select])[condition3]
						condition4 = x_select > np.argmax(x_select2)-2
						x_select2  = np.select([condition4],[x_select2])[condition4]
						y_select2  = np.select([condition4],[y_select2])[condition4]
					except ValueError:
						pass
				try:
					popt3, pcov3 = curve_fit(gaus, x_select2, y_select2)
					if np.absolute(best_shift-popt3[1]) < replace_shift_criteria:
						best_shift = popt3[1]
					if test:
						ax3.plot(x_select2, gaus(x_select2, popt3[0], popt3[1], popt3[2]),
							     color='salmon',
							     label="gaussian fit for selected parameters, gaussian fitted pixel:{} ($\AA$)".format(round(popt3[1],5)),
							     alpha=0.5)
						ax3.axvline(x=popt3[1], linewidth=0.5, linestyle='--', color='salmon')
				
				except RuntimeError:
					pass
				
				except TypeError:
					pass
		
		except RuntimeError:
			pass
		
		except TypeError:
			pass

		except ValueError: #NaNs
			pass
		
	if test:
		ax3.plot(np.arange(-delta_wave_range, delta_wave_range+step, step), xcorr_list,
			     color='black', label='cross correlation', alpha=0.5)
		ax3.plot([best_shift, best_shift], [float(np.min(xcorr_list)),
			     float(np.max(xcorr_list))], 'k:',
			     label="best wavelength shift:{} ($\AA$)".format(\
				 round(best_shift,5)))
		ax3major_ticks = np.arange(-(delta_wave_range), (delta_wave_range+1), 2)
		ax3minor_ticks = np.arange(-(delta_wave_range), (delta_wave_range+1), 0.1)
		ax3.set_xticks(ax3major_ticks)
		ax3.set_xticks(ax3minor_ticks, minor=True)
		#ax3.set_title("Cross-Correlation Plot, pixels start at {} with width {}"\
		#	.format(start_pixel, window_width))
		ax3.set_xlabel("Wavelength shift ($\AA$)")
		ax3.set_ylabel('Cross correlation')
		ax3.set_xlim(-(delta_wave_range), (delta_wave_range))
		ax3.set_ylim(np.min(xcorr_list), np.max(xcorr_list))
		ax3.legend()
		#ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
		#plt.tight_layout(h_pad=3.3)
		if testname is None:
			testname = 'test'
		plt.savefig('{}_{}.png'.format(testname,start_pixel))#), bbox_inches='tight')
		plt.close()

	return best_shift



def wavelengthSolutionFit(data, model, order, **kwargs):
	"""
	Using the least-square method in fitting a 4th order polynomial 
	to obtain a new wavelength solution.
	
	Parameters
	----------
	data 		:  	spectrum object
	       			The input telluric data 

	model 		: 	model object
		   			The telluric model to calculate the delta wavelength

	order		: 	int
				  	spectral order defined by NIRSPEC

	Optional Parameters
	-------------------
	window_width:	int
				 	Width of each xcorr window
				  	Default is 40

	window_step : 	int
				  	Step size between adjacent xcorr windows
				  	Default is 5

	xcorr_range : 	float
				  	Max shift value for each xcorr window in 
				  	WAVELENGTH (Angstrom)
				  	Default is 15 Angstroms


	xcorr_step 	: 	float
				  	Shift step size for each xcorr window
				  	The xcorr calculation ranges from -xcorr_range to
				  	+xcorr_range in steps of this value
				  	Default is 0.05 Angstroms

	niter		: 	int
				  	Number of interations for fitting the wavelength solution
				  	Default is 15

	outlier_rej : 	float
					The number of sigmas in fitting the wavelength solution
					Default is 2

	test 		: 	boolean
					Output the diagnostic plots for each xcorr value
					This option will increase the total calculation time
					Default is False

	save 		: 	boolean
					Save the resulting wavelength solution as a new fits file
					The new file name would be "_new_wave_sol_order" after
					the original name
					The new wavelength solution parameters will be included in
					the corresponding fits header
					Default is False

	save_to_path: 	str
					This option is to specify the path to save the 
					new wavelength solution fits file. 
					The new wavelength solution telluric data will be 
					saved as its original name + 
					'_new_wave_sol_{order}_all.fits'
					Default is None

	data_path 	:	str
					The option specifies the path to the telluric data.
					If it is not specified, this will be the same path as
					of the save_to_path option.
					Default is None


	Returns
	-------
	new_wave_sol: 	numpy array
				    a new wavelength solution

	p0 			:	numpy array
					Fitted wavelength solution paramters

	width_range_center 	: numpy array
					the pixel center array for wavelength fitting

	residual 	: 	numpy array
					the residual of each fitting wavelength 
					correction and the wavelength shift
	
	#std 		:	float
	#				standard deviation for the final fit in Angstrom
	
	#stdV 		: 	float
	#				standard deviation for the final fit in km/s

	"""
	# set up the initial parameters
	#spec_range = kwargs.get('spec_range',900)
	#order = kwargs.get('order', None)
	width              = kwargs.get('window_width', 40)
	step_size          = kwargs.get('window_step', 5)
	delta_wave_range   = kwargs.get('xcorr_range', 15)
	step               = kwargs.get('xcorr_step', 0.05)
	niter              = kwargs.get('niter', 15)
	outlier_rej        = kwargs.get('outlier_rej', 3)
	apply_sigma_mask   = kwargs.get('apply_sigma_mask', False) # apply a simple outlier rejection mask
	mask_custom        = kwargs.get('mask_custom', []) # apply a simple outlier rejection mask
	test               = kwargs.get('test', False) # output the xcorr plots
	save               = kwargs.get('save', False) # save the new wavelength solution
	save_to_path       = kwargs.get('save_to_path', None)
	data_path          = kwargs.get('data_path', None)
	length1            = kwargs.get('length1', len(data.oriWave)) # length of the input array
	# calculation the necessary parameters
	pixel_range_start  = kwargs.get('pixel_range_start',0)
	pixel_range_end    = kwargs.get('pixel_range_end',-1)
	pixel0             = np.delete(np.arange(length1), np.union1d(data.mask, mask_custom))
	#pixel0             = np.arange(length1)
	pixel              = pixel0[pixel_range_start:pixel_range_end]

	## Ultimately this should be removed by the airmass parameter
	# increase the telluric model strength for N3
	if order == 63 or order == 64 or order == 65 or order == 66:
		model.flux **= 6
	elif order == 62:
		model.flux **= 4
	elif order == 59:
		model.flux **= 3
	elif order == 61:
		model.flux **= 2
	elif order == 35:
		model.flux **= 2
	elif order == 36:
		model.flux **= 3
	elif order == 37:
		model.flux **= 1.2

	# LSF of the intrument
	vbroad = (299792.458)*np.mean(np.diff(data.wave))/np.mean(data.wave)
	#vbroad = smart.getLSF(data2, continuum=False)
	print("LSF for telluric wavelength calibration: ", vbroad)

	## test
	#for index in [189,  320,  449,  498,  941]:
	#	index = int(index)
	#	index_left  = index - 1
	#	index_right = index + 1
	#	data.flux[index] = (data.flux[index_left] + data.flux[index_right])/2
	data2       = copy.deepcopy(data)
	model2      = copy.deepcopy(model)

	# apply the custom mask
	#plt.figure()
	#plt.plot(data2.wave, data2.flux, lw=0.5, alpha=0.5, c='b')
	#if apply_sigma_mask:
	#	mask_combined = np.union1d(mask_custom, data.mask)
	#	print(mask_combined, type(mask_combined))
	#	for i in mask_combined:
	#		if (int(i) > pixel_range_start) and (int(i) < length1 + pixel_range_end -1): 
	#			data2.oriFlux[int(i)] = (data2.oriFlux[int(i)-1] + data2.oriFlux[int(i)+1])/2
	#			data2.flux = data2.oriFlux
	#			data2.wave = data2.oriWave

	#	data2.flux = data2.flux[pixel_range_start:pixel_range_end]
	#	data2.wave = data2.wave[pixel_range_start:pixel_range_end]


	#	plt.plot(data.wave, data.flux, 'k-', alpha=0.5, label='original data')
	#	plt.plot(data2.wave, data2.flux, 'r-', alpha=0.5, label='median combined mask data')
	#	plt.xlabel('$\lambda$ ($\AA)')
	#	plt.ylabel('$F_{\lambda}$')
	#	plt.legend()
	#	plt.show()
	#	plt.close()
	#	sys.exit()
	#	#pixel0      = np.delete(pixel0, mask_combined)
	#	#data2.wave  = np.delete(data.wave, mask_combined)
	#	#data2.flux  = np.delete(data.flux, mask_combined)

	#plt.plot(data2.oriWave, data2.oriFlux/200.0, linewidth=0.5, alpha=0.5, color='black')
	#plt.plot(data2.wave, data2.flux, linewidth=0.5, alpha=0.5, color='red')
	#plt.show()
	#plt.close()
	#sys.exit()

	model2.flux = smart.broaden(wave=model2.wave, flux=model2.flux, vbroad=vbroad, 
		                      rotate=False, gaussian=True)
	modelCC     = copy.deepcopy(model2) # Use this for final CC
	# model resample and LSF broadening
	model2.flux = np.array(smart.integralResample(xh=model2.wave, 
		                                        yh=model2.flux, xl=data2.wave))
	model2.wave = data2.wave

	# fitting the new wavelength solution
	#if test is True:
	#	print("initial WFIT:",data2.header['WFIT0'],data2.header['WFIT1'],
	#		data2.header['WFIT2'],data2.header['WFIT3'],data2.header['WFIT4'],
	#		data2.header['WFIT5'])

	time0 = time.time()

	for i in range(niter):
	# getting the parameters of initial wavelength solution 

		#if i > 4: break

		if i == 0: # Change the width for the first iteration
			width     = 300
			step_size = 10
			include_ends = True #False
		elif i == 1: # Change the width for the second iteration
			width     = 150
			step_size = 10
			include_ends = True
		elif i == 2: # Change the width for the second iteration
			width     = 100
			step_size = 10
		else: # Change the width for the middle few iterations
			width     = 80
			step_size = 10

		if include_ends:
			spec_range          = len(pixel) # window range coverage for xcorr
			endwidth            = 40
			#print(spec_range-endwidth//2)

			#x1 = np.arange(pixel_range_start, width//2-endwidth//2, step_size)
			#x2 = np.arange(pixel_range_start, spec_range+pixel_range_start-width, step_size)
			#x3 = np.arange(x2[-1]+width//2, spec_range-endwidth//2, step_size)
			#width_ranges        = np.concatenate( [ x1, x2, x3 ] )
			#width_range_centers = np.concatenate( [ x1 + endwidth//2, x2 + width//2, x3 + endwidth//2 ] )
			#widths              = np.concatenate( [ np.zeros(len(x1), dtype=np.int) + endwidth,
			#	  								    np.zeros(len(x2), dtype=np.int) + width,
			#	 								    np.zeros(len(x3), dtype=np.int) + endwidth ] )

			x1 = np.arange(0, width//2-endwidth, step_size)
			x2 = np.arange(0, spec_range-width, step_size)
			x3 = np.arange(x2[-1]+width//2, spec_range-endwidth, step_size)
			width_ranges        = np.concatenate( [ x1, x2, x3 ] )
			width_range_centers = np.concatenate( [ x1 + endwidth, x2 + width//2, x3 + endwidth ] )
			widths              = np.concatenate( [ np.zeros(len(x1), dtype=np.int) + endwidth,
				  								    np.zeros(len(x2), dtype=np.int) + width,
				 								    np.zeros(len(x3), dtype=np.int) + endwidth ] )

			#print(i, width, width//2, spec_range+pixel_range_start-width//2, spec_range)
			#print(spec_range+pixel_range_start)
			#print(i, width, width//2, spec_range-width, spec_range)		
			#print(spec_range+pixel_range_start)
			#print(width_ranges)
			#print(width_range_centers)
			#print(widths)
			#sys.exit()
		else:
			spec_range          = len(pixel) # window range coverage for xcorr
			width_ranges        = np.arange(pixel_range_start, spec_range+pixel_range_start-width, step_size)
			width_range_centers = np.arange(pixel_range_start, spec_range+pixel_range_start-width, step_size) + width//2
			widths              = np.zeros(len(width_ranges), dtype=np.int) + width
			#print(i, width, spec_range)
			#print(width_range_centers)
			#print(widths)
			#sys.exit()

		time1 = time.time()

		if i is 0:
			wfit0 = data2.header['WFIT0']
			wfit1 = data2.header['WFIT1']
			wfit2 = data2.header['WFIT2']
			wfit3 = data2.header['WFIT3']
			wfit4 = data2.header['WFIT4']
			wfit5 = data2.header['WFIT5']
			c3    = 0
			c4    = 0
			p0    = np.array([wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4])

		else:
			wfit0     = data2.header['WFIT0NEW']
			wfit1     = data2.header['WFIT1NEW']
			wfit2     = data2.header['WFIT2NEW']
			wfit3     = data2.header['WFIT3NEW']
			wfit4     = data2.header['WFIT4NEW']
			wfit5     = data2.header['WFIT5NEW']
			c3        = data2.header['C3']
			c4        = data2.header['C4']
			popt0_ori = data2.header['POPT0']
			popt1_ori = data2.header['POPT1']
			popt2_ori = data2.header['POPT2']
			popt3_ori = data2.header['POPT3']
			popt4_ori = data2.header['POPT4']
			popt5_ori = data2.header['POPT5']
			popt6_ori = data2.header['POPT6']
			popt7_ori = data2.header['POPT7']
			p0        = np.array([wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4])

		# calcutate the delta wavelentgh
		best_shift_list  = []
		#for counter, j, width in enumerate(width_range):
		#print('TEST0', len(widths), len(width_ranges), len(width_range_centers))
		for counter, j, center, width in zip(range(len(widths)), width_ranges, width_range_centers, widths):
			testname = "loop{}".format(i+1)
			if i == 0:
				#print(counter, j, center, width)
				time2 = time.time()
				best_shift = smart.pixelWaveShift(data2, model, j, width, delta_wave_range, model2,
												test=test, testname=testname,
					                            counter=counter, step=step,
					                            pixel_range_start=pixel_range_start,
					                            pixel_range_end=pixel_range_end,
					                            lsf=vbroad, length1=length1, pixel=pixel0)
				time3 = time.time()
				if test is True:
					print("xcorr time: {} s".format(round(time3-time2, 4)))
			elif i == 1:
				if delta_wave_range >= 5:
					# reduce the delta_wave_range as 5
					delta_wave_range = 5
					time2 = time.time()
					best_shift = smart.pixelWaveShift(data3, model, j, width, delta_wave_range, model2,
													test=test, testname=testname,
													counter=counter, step=step,
													pixel_range_start=pixel_range_start,
													pixel_range_end=pixel_range_end,
													lsf=vbroad, length1=length1, pixel=pixel0)
					time3 = time.time()
					if test is True:
						print("xcorr time: {} s".format(round(time3-time2, 4)))
				elif delta_wave_range < 5:
					time2 = time.time()
					delta_wave_range = 2
					best_shift = smart.pixelWaveShift(data3, model, j, width, delta_wave_range, model2,
													test=test, testname=testname,
													counter=counter, step=step,
													pixel_range_start=pixel_range_start,
													pixel_range_end=pixel_range_end,
													lsf=vbroad, length1=length1, pixel=pixel0)
					time3 = time.time()
					if test is True:
						print("xcorr time: {} s".format(round(time3-time2, 4)))
			else:
				# reduce the delta_wave_range as 2
				time2 = time.time()
				if delta_wave_range > 2:
					delta_wave_range = 0.6
				if i > 4:
					step = 0.01
				best_shift = smart.pixelWaveShift(data3, model, j, width, delta_wave_range, model2,
											    test=test, testname=testname,
											    counter=counter, step=step,
												pixel_range_start=pixel_range_start,
												pixel_range_end=pixel_range_end,
												lsf=vbroad, length1=length1, pixel=pixel0)
				time3 = time.time()
				if test is True:
					print("xcorr time: {} s".format(round(time3-time2, 4)))


			#print(counter, center, best_shift)
			#if counter == 0:
			#	print(best_shift_list)
			best_shift_list.append(best_shift)
			#print(len(best_shift_list))

			time4 = time.time()
		if test is True:
			print("Total X correlation time for loop {}: {} s".format(i+1, round(time4-time1,4)))
		print("Total X correlation time for loop {}: {} s".format(i+1, round(time4-time1, 4)))
		
		# fit a new wavelength solution
		def waveSolutionFn0(orderNum):
			def fitFn(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5):
				order    = float(orderNum)
				wave_sol = wfit0 + wfit1*pixel + wfit2*pixel**2 + \
				           (wfit3 + wfit4*pixel + wfit5*pixel**2)/order 

				return wave_sol
			return fitFn

		def waveSolutionFn1(orderNum):
			def fitFn(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4):
				order    = float(orderNum)
				wave_sol = wfit0 + wfit1*pixel + wfit2*pixel**2 + \
				           (wfit3 + wfit4*pixel + wfit5*pixel**2)/order + \
				           c3*pixel**3 + c4*pixel**4

				return wave_sol
			return fitFn

		# fit a low order polynomial for the first iteration
		best_shift_list2 = np.asarray(best_shift_list)
		
		# We don't want values that didn't converge
		mask1            = np.where(abs(best_shift_list2) != delta_wave_range) 

		if i==0:
			p1 = p0[:-2]
			popt, pcov = curve_fit(waveSolutionFn0(order), width_range_centers[mask1], 
				                   best_shift_list2[mask1], p1)
			popt = np.append(popt,[0,0])

		# fit a new wavelength solution
		# fit a fourth order polynomial for later iterations
		else:
			popt, pcov = curve_fit(waveSolutionFn1(order), width_range_centers[mask1], 
				                   best_shift_list2[mask1], p0)

		# outlier rejection
		best_shift_array = best_shift_list2[mask1]
		
		m = outlier_rej # number of sigma for outlier rejection
		if i == 0:
			fit_sigma = 2#0.8
			#fit_sigma = np.std(original_fit - best_shift_array)
			original_fit = smart.waveSolution(width_range_centers[mask1], *popt, order=order)
		elif i < 5:
			fit_sigma = data2.header['FITSTD']
			m = 3. # Change the sigma!
			#if i+1 == 4: m = 2. # Change the sigma!
			original_fit = smart.waveSolution(width_range_centers[mask1], *popt, order=order)
		else:
			fit_sigma = data2.header['FITSTD']
			#m = 1.5 # Change the sigma!
			original_fit = smart.waveSolution(width_range_centers[mask1], 
											popt0_ori, popt1_ori, popt2_ori, popt3_ori,
				                            popt4_ori, popt5_ori, popt6_ori, popt7_ori,
				                            order=order)

		# exclude the edge pixels in the fitting
		#width_range_center2 = width_range_center[5:-5]
		#best_shift_array2 = best_shift_array[5:-5]
		#width_range_center2 = width_range_center2[np.where(\
		#	abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		#best_shift_array2 = best_shift_array2[np.where(\
		#	abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		#if order == 60 and k is 1:
		#	print("use outlier rejection factor of 1 for order 60 in the first iteration")
		#	m = 1
		width_range_center2 = width_range_centers[mask1][np.where \
		                                         (abs(original_fit - best_shift_array) < m*fit_sigma)]
		
		best_shift_array2   = best_shift_array[np.where \
		                                       (abs(original_fit - best_shift_array) < m*fit_sigma)]		

		if len(width_range_center2) < 8:
			print("Number of selected pixel < number of fits parameters (8)")
			#width_range_center2 = width_range_center
			#best_shift_array2   = best_shift_array
			if i != 0:
				residual2           = smart.waveSolution(width_range_centers[mask1], 
													   popt0_ori, popt1_ori, popt2_ori, popt3_ori,
					                                   popt4_ori, popt5_ori, popt6_ori, popt7_ori, 
					                                   order=order) - best_shift_array
				residual2           = residual2[np.where(abs(original_fit - best_shift_array) < m*fit_sigma)]

         
				variance2           = ((residual2 ** 2).sum()) / (len(residual2) - 1)
				RMSE2               = np.sqrt(variance2)
				break

		elif len(width_range_center2) < len(width_range_centers[mask1])*0.4 and i != 0:
			print("The iteration stops because the selected points for fitting",
				len(width_range_center2),"are smaller than 2/5 of the total points",
				len(width_range_centers[mask1]))
			#width_range_center2 = width_range_center
			#best_shift_array2   = best_shift_array
			residual2           = smart.waveSolution(width_range_centers[mask1], 
												   popt0_ori, popt1_ori, popt2_ori, popt3_ori,
				                                   popt4_ori, popt5_ori, popt6_ori, popt7_ori, 
				                                   order=order) - best_shift_array
			residual2           = residual2[np.where \
			                                (abs(original_fit - best_shift_array) < m*fit_sigma)]

			variance2           = ((residual2 ** 2).sum()) / (len(residual2) - 1)
			RMSE2               = np.sqrt(variance2)
			break

		# fit the wavelength again after the outlier rejections
		if i==0:
			p1 = p0[:-2]
			popt2, pcov2  = curve_fit(waveSolutionFn0(order),
				                      width_range_center2, best_shift_array2, p1)
			popt2         = np.append(popt2, [0,0])
			#if i==0: m=1.2
			for num_fit in range(8):
				## re-fit for five times after the second outlier rejection
				## the residual fit in the first iteration is the most important part
				original_fit2       = smart.waveSolution(width_range_centers[mask1], *popt2, order=order)
				width_range_center2 = width_range_centers[mask1][np.where(abs(original_fit2 - best_shift_array) < m*fit_sigma)]
				best_shift_array2   = best_shift_array[np.where(abs(original_fit2 - best_shift_array) < m*fit_sigma)]	
				popt2, pcov2        = curve_fit(waveSolutionFn0(order), width_range_center2, best_shift_array2, p1)
				popt2               = np.append(popt2, [0,0])

		else:
			popt2, pcov2 = curve_fit(waveSolutionFn1(order),
				                     width_range_center2, best_shift_array2, p0)

			for num_fit in range(5):
				## re-fit again after the second outlier rejection
				original_fit2       = smart.waveSolution(width_range_centers[mask1], *popt2, order=order)
				width_range_center2 = width_range_centers[mask1][np.where(abs(original_fit2 - best_shift_array) < m*fit_sigma)]
				best_shift_array2   = best_shift_array[np.where(abs(original_fit2 - best_shift_array) < m*fit_sigma)]	
				
				if len(width_range_center2) > 8:
					popt2, pcov2        = curve_fit(waveSolutionFn1(order), width_range_center2, best_shift_array2, p0)
				
				else:
					print("The iteration stops because the selected points for fitting",
						len(width_range_center2),"are fewer than number of parameters = 8")
					popt2               = popt2previous
					width_range_center2 = width_range_centers[mask1]
					best_shift_array2   = best_shift_array
					break

				if len(width_range_center2) < len(width_range_centers[mask1])*0.4 and i != 0:
					print("The iteration stops because the selected points for fitting",
						len(width_range_center2),"are smaller than 2/5 of the total points",
						len(width_range_centers[mask1]))
					popt2 = popt2previous
					break


		# update the parameters
		wfit0 += popt2[0]
		wfit1 += popt2[1]
		wfit2 += popt2[2]
		wfit3 += popt2[3]
		wfit4 += popt2[4]
		wfit5 += popt2[5]
		c3    += popt2[6]
		c4    += popt2[7]
		p0    = np.array([wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4])
		#if test is True:
		#	print("fitted p0: ",p0)
		
		# update the fits header keywords WFIT0-5, c3, c4
		data2.header['COMMENT']  = 'Below are the keywords added by NIRSPEC_PIP...'
		data2.header['WFIT0NEW'] = wfit0
		data2.header['WFIT1NEW'] = wfit1
		data2.header['WFIT2NEW'] = wfit2
		data2.header['WFIT3NEW'] = wfit3
		data2.header['WFIT4NEW'] = wfit4
		data2.header['WFIT5NEW'] = wfit5
		data2.header['c3']       = c3
		data2.header['c4']       = c4
		if i == 0:
			data2.bestshift      = np.asarray(best_shift_list)
		#else:
			#data2.bestshift      = data2.bestshift + np.asarray(best_shift_list)
		data2.header['FITSTD']   = np.std(smart.waveSolution(width_range_center2, *popt2, 
			                                               order=order) - best_shift_array2)
		data2.header['POPT0']	 = popt2[0]
		data2.header['POPT1']	 = popt2[1]
		data2.header['POPT2']	 = popt2[2]
		data2.header['POPT3']	 = popt2[3]
		data2.header['POPT4']	 = popt2[4]
		data2.header['POPT5']	 = popt2[5]
		data2.header['POPT6']	 = popt2[6]
		data2.header['POPT7']	 = popt2[7]

		new_wave_sol  = smart.waveSolution(pixel,  wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4, order=order)
		new_wave_sol0 = smart.waveSolution(pixel0, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4, order=order)

		time5 = time.time()
		if test:
			print("Pixel wavelength fit time for loop {}: {} s".format(i+1, round(time5-time4, 4)))
		## plot for analysis
		data3       = copy.deepcopy(data)
		#data3.wave  = new_wave_sol0
		data3.wave  = new_wave_sol
		model3      = copy.deepcopy(model)
		model3.flux = smart.broaden(wave=model3.wave, flux=model3.flux, vbroad=vbroad, 
			                      rotate=False, gaussian=True)
		# model resample and LSF broadening
		model3.flux = np.array(smart.integralResample(xh=model3.wave, 
			                                        yh=model3.flux, xl=data3.wave))
		model3.wave = data3.wave

		#plt.rc('text', usetex=True)
		#plt.rc('font', family='sans-serif')
		fig = plt.figure(figsize=(14,8))
		gs1 = gridspec.GridSpec(5, 4)
		ax1 = plt.subplot(gs1[0:2,0:])
		ax2 = plt.subplot(gs1[2:4,0:])
		ax3 = plt.subplot(gs1[4:,0:], sharex=ax2)
		
		ax1.xaxis.tick_top()
		#ax1.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=0.8)
		if not apply_sigma_mask:
			#ax1.plot(new_wave_sol, data.flux[pixel], color='black', linestyle='-', 
			#		 label='corrected telluric data', alpha=1, linewidth=0.5)
			ax1.plot(new_wave_sol0, data3.flux, color='black', linestyle='-', 
					 label='corrected telluric data', alpha=1, linewidth=0.5)
		else:
			#ax1.plot(new_wave_sol, data.flux[pixel], color='black', linestyle='-', 
			#		 label='corrected telluric data', alpha=1, linewidth=0.5)
			ax1.plot(new_wave_sol, data3.flux, color='black', linestyle='-', 
					 label='corrected telluric data', alpha=1, linewidth=0.5)
		ax1.plot(model3.wave, model3.flux, 'r-' , label='telluric model', alpha=0.7, lw=0.5)
		ax1.set_xlabel("Wavelength ($\AA$)")
		ax1.set_ylabel('Transmission')
		ax1.xaxis.set_label_position('top') 
		ax1.set_xlim(new_wave_sol[0], new_wave_sol[-1])
		ax1.get_xaxis().get_major_formatter().set_scientific(False)
		ax1.legend(frameon=False)

		residual1 = smart.waveSolution(width_range_centers[mask1], *popt, 
			                         order=order) - best_shift_array
		residual2 = smart.waveSolution(width_range_center2, *popt2, 
			                         order=order) - best_shift_array2
		variance2 = ((residual2 ** 2).sum()) / (len(residual2) - 1)
		RMSE2     = np.sqrt(variance2)
		std       = round(np.std(residual2), 4)	

		ax2.plot(width_range_centers, best_shift_list, 'k.', label="delta wavelength")
		ax2.plot(width_range_center2, best_shift_array2, 'b.',
			     label="delta wavelength with outlier rejection")
		#ax2.plot(width_range_center,smart.waveSolution(width_range_center,
		#	popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],
		#	order=order),'g.',label="fitted wavelength function".format(np.std(residual1)),alpha=0.5)
		ax2.plot(width_range_center2, smart.waveSolution(width_range_center2, *popt2, order=order), 'r-',
			     label="fitted wavelength function with outlier rejection, STD = {} $\AA$ ({} km/s), RMS = {} $\AA$ ({} km/s)".format(\
			     np.round_(std, decimals=3), np.round_(std/np.average(new_wave_sol)*299792.458, decimals=3),
			     np.round_(RMSE2, decimals=3), np.round_(RMSE2/np.average(new_wave_sol)*299792.458, decimals=3)),
			     alpha=0.5)
		ax2.axhline(-delta_wave_range, c='r', ls=':')
		ax2.axhline(delta_wave_range, c='r', ls=':')
		ax2.set_ylabel(r"$\Delta$ $\lambda$ ($\AA$)")
		ax2.set_xlim(pixel[0], pixel[-1])
		ax2.legend(frameon=False)
		# plot the residual
		plt.setp(ax2.get_xticklabels(), visible=False)

		yticks = ax3.yaxis.get_major_ticks()
		yticks[-1].label1.set_visible(False)
		#ax3.plot(width_range_centers,residual1,'g.',alpha=0.5)
		ax3.plot(width_range_center2, residual2, 'r.', alpha=0.5)
		ax3.set_ylim(-3*fit_sigma, 3*fit_sigma)
		ax3.set_ylabel("Residual ($\AA$)")
		ax3.set_xlabel('Pixel')
		ax3.set_xlim(pixel[0], pixel[-1])
		#ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.5))

		ax1.minorticks_on()
		ax2.minorticks_on()
		ax3.minorticks_on()

		plt.subplots_adjust(hspace=.0)
		plt.savefig("pixel_to_delta_wavelength_loop_{}.png".format(i+1),
					bbox_inches='tight')
		plt.close()

		time6 = time.time()
		if test is True:
			print("Plot time:",format(round(time6-time5, 4)))
		
		if data2.header['FITSTD']>fit_sigma and i+1 > 4:
			print("Wavelength solution converges in {} loops, with STD: {} Angstrom ({} km/s)".format(i+1,
				  np.std(residualprevious), np.std(residualprevious)/np.average(new_wave_sol)*299792.458))
			print("RMS: {} Angstrom ({} km/s)".format(RMSEprevious, RMSEprevious/np.average(new_wave_sol)*299792.458))
			print("Total calculation time: {} min".format(round((time6-time0)/60., 4)))
			break

		elif len(width_range_center2) < len(width_range_centers[mask1])*0.4 and i != 0:
			print("The iteration stops because the selected points for fitting",
				  len(width_range_center2),"are smaller than 2/5 of the total points",
				  len(width_range_centers[mask1]))
			print("Wavelength solution converges in {} loops, with STD: {} Angstrom ({} km/s)".format(i+1,
				  np.std(residualprevious), np.std(residualprevious)/np.average(new_wave_sol)*299792.458))
			print("RMS: {} Angstrom ({} km/s)".format(RMSEprevious, RMSEprevious/np.average(new_wave_sol)*299792.458))
			print("Total calculation time: {} min".format(round((time6-time0)/60., 4)))
			break

		else: # save previous values in case we break early
			popt2previous               = popt2
			residualprevious            = residual2 
			width_range_center_previous = width_range_center2
			previousModel				= model3
			RMSEprevious                = RMSE2


	if save is True:
		if data_path is None:
			data_path = save_to_path + '_' + str(order) + '_all.fits'
		save_name = save_to_path + "_calibrated_{}_all.fits".format(order)
		with fits.open(data_path) as hdulist:
			hdulist[0].header['COMMENT']  = 'Below are the keywords added by NIRSPEC_PIP...'
			hdulist[0].header['WFIT0NEW'] = wfit0
			hdulist[0].header['WFIT1NEW'] = wfit1
			hdulist[0].header['WFIT2NEW'] = wfit2
			hdulist[0].header['WFIT3NEW'] = wfit3
			hdulist[0].header['WFIT4NEW'] = wfit4
			hdulist[0].header['WFIT5NEW'] = wfit5
			hdulist[0].header['c3']       = c3
			hdulist[0].header['c4']       = c4
			#hdulist[0].bestshift          = data2.bestshift + best_shift_list
			hdulist[0].header['FITSTD']   = np.std(smart.waveSolution(\
			       								   width_range_center2, *popt2,
												   order=order) - best_shift_array2)  ### XXX THIS NEEDS TO BE FIXED!
			hdulist[0].header['POPT0']	  = popt2[0]	### XXX DO WE REALLY NEED ALL THESE? THEY ARE ONLY THE LAST ITERATION!
			hdulist[0].header['POPT1']	  = popt2[1]
			hdulist[0].header['POPT2']	  = popt2[2]
			hdulist[0].header['POPT3']	  = popt2[3]
			hdulist[0].header['POPT4']	  = popt2[4]
			hdulist[0].header['POPT5']	  = popt2[5]
			hdulist[0].header['POPT6']	  = popt2[6]
			hdulist[0].header['POPT7']	  = popt2[7]
			hdulist[0].header['STD']      = str(np.std\
				                               (residualprevious)/np.average(new_wave_sol)*299792.458) + 'km/s'
			hdulist[0].header['RMS']      = str(RMSEprevious/np.average(new_wave_sol)*299792.458) + 'km/s'
			hdulist[0].data               = smart.waveSolution(np.arange(length1),
				                                             wfit0, wfit1, wfit2, wfit3, 
				                                             wfit4, wfit5, c3, c4, order=order)
			try:
				hdulist.writeto(save_name, overwrite=True)
			except FileNotFoundError:
				hdulist.writeto(save_name)
		print("The new wavelength solution file is saved to {}".format(save_name))

	std  = np.std(residualprevious)
	stdV = np.std(residualprevious)/np.average(new_wave_sol)*299792.458

	return new_wave_sol, p0, width_range_center_previous, residualprevious, best_shift_list



def run_wave_cal(data_name, data_path, order_list,
	             save_to_path, test=False, save=False, plot_masked=False,
	             window_width=40, window_step=5, mask_custom=[], apply_sigma_mask=False, apply_edge_mask=False, pwv='1.5',
	             xcorr_step=0.05, niter=20, outlier_rej=None, defringe_list=[62], cal_param=None):
	"""
	Run the telluric wavelength calibration.

	Parameters
	----------

	"""

	##################################
	## parameters set up
	##################################
	#xcorr_step    = 0.05
	#niter         = 20
	#outlier_rej   = 3.
	#airmass       = '1.5'
	#pwv           = '0.5'
	#defringe_list = [62]
	#apply_sigma_mask = apply_sigma_mask # if True: apply a simple mask
	##################################
	print('mask_custom', mask_custom)

	original_path = os.getcwd()

	for order in order_list:
		print("Start telluric wavelength calibration on {} order {}".format(data_name,order))

		# load the default calibration parameters
		if cal_param is not None:
			xcorr_range       = cal_param[str(order)]['xcorr_range']
			pixel_range_start = cal_param[str(order)]['pixel_range_start']
			pixel_range_end   = cal_param[str(order)]['pixel_range_end']
			if outlier_rej is None:
				if cal_param[str(order)]['outlier_rej'] is not None:
					outlier_rej = cal_param[str(order)]['outlier_rej']
				else:
					outlier_rej = 3.
		else:
			xcorr_range       = cal_param_nirspec[str(order)]['xcorr_range']
			pixel_range_start = cal_param_nirspec[str(order)]['pixel_range_start']
			pixel_range_end   = cal_param_nirspec[str(order)]['pixel_range_end']
			if outlier_rej is None:
				outlier_rej = cal_param_nirspec[str(order)]['outlier_rej']

		if pixel_range_end == -1 and apply_sigma_mask is False:
			pixel_range_end   += -25

		directory = save_to_path + '/O{}'.format(order)
		# create a new directory and change to the new dir
		if not os.path.exists(directory):
			os.makedirs(directory)
		os.chdir(directory)

		# use median value to replace the masked values later
		data     = smart.Spectrum(name=data_name, order=order, path=data_path, apply_sigma_mask=apply_sigma_mask)
		length1  = len(data.oriWave) # preserve the length of the array

		# the telluric standard model
		wavelow  = data.wave[0]  - 200
		wavehigh = data.wave[-1] + 200
		# read the airmass from the data
		airmass  = float(round(data.header['AIRMASS']*2)/2)
		if airmass > 3.0: airmass = 3.0
		print('airmass', airmass)
		airmass  = str(airmass)
		
		"""
		## testing
		mask_combined = [2, 3, 497, 498, 499, 500, 501, 502, 503, 685]
		for i in mask_combined:
			data.flux[int(i)] = (data.flux[int(i)-1] + data.flux[int(i)+1])/2
		data.flux[496] = 87
		data.flux[497] = 86.5
		data.flux[498] = 86
		data.flux[499] = 85.5
		data.flux[500] = 85
		data.flux[501] = 85.5
		data.flux[502] = 84
		data.flux[503] = 84.5
		"""

		## estimating the pwv parameter
		#pwv_list = ['0.5', '1.0', '1.5', '2.5', '3.5', '5.0', '7.5', '10.0', '20.0']
		#pwv_chi2 = []
		#for pwv in pwv_list:
		#	data_tmp       = copy.deepcopy(data)

		#	data_tmp.flux  = data_tmp.flux[pixel_range_start:pixel_range_end]
		#	data_tmp.wave  = data_tmp.wave[pixel_range_start:pixel_range_end]
		#	data_tmp.noise = data_tmp.noise[pixel_range_start:pixel_range_end]

		#	model_tmp      = smart.getTelluric(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)
		#	data_tmp       = smart.continuumTelluric(data=data_tmp, model=model_tmp)

		#	model_tmp.flux = np.array(smart.integralResample(xh=model_tmp.wave, yh=model_tmp.flux, xl=data_tmp.wave))
		#	model_tmp.wave = data_tmp.wave

		#	pwv_chi2.append(smart.chisquare(data_tmp, model_tmp))
		# find the pwv with minimum chisquare
		#pwv_chi2_array = np.array(pwv_chi2)
		#
		#plt.plot(pwv_list, pwv_chi2)
		#plt.xlabel('pwv (mm)', fontsize=15)
		#plt.ylabel('$\chi^2$', fontsize=15)
		#plt.tight_layout()
		#plt.savefig(save_to_path+'/O{}/pwv_chi2.png'.format(order))
		##plt.show()
		#plt.close()
		##sys.exit()

		#pwv_min_index = np.where(pwv_chi2_array == np.min(pwv_chi2_array))[0][0]
		#pwv           = pwv_list[pwv_min_index]
		#pwv = '1.5' #self defined pwv
		print('pwv', pwv)
		

		model    = smart.getTelluric(wavelow=wavelow, wavehigh=wavehigh, airmass=airmass, pwv=pwv)

		## Use the median values to replace bad single pixels
		data0 = copy.deepcopy(data)

		# this is to apply a sigma clipping mask
		if apply_sigma_mask:
			#data.flux  = data.oriFlux
			#data.wave  = data.oriWave
			#data.noise = data.oriNoise
			
			mask_combined = np.union1d(mask_custom, data.mask)

			print('mediam-averaging the pixels:', data.mask)

			#for i in mask_combined:
			#	if (int(i) > pixel_range_start) and (int(i) < length1 + pixel_range_end -1): 
			#		data.oriFlux[int(i)] = (data.oriFlux[int(i)-1] + data.oriFlux[int(i)+1])/2
			#		data.flux = data.oriFlux

			# select the pixel for wavelength calibration
			data.flux  = data.flux[pixel_range_start:pixel_range_end]
			data.wave  = data.wave[pixel_range_start:pixel_range_end]
			data.noise = data.noise[pixel_range_start:pixel_range_end]
		elif not apply_sigma_mask:
			if apply_edge_mask:
				data.flux  = np.delete(data.oriFlux, mask_custom)[pixel_range_start: pixel_range_end]
				data.wave  = np.delete(data.oriWave, mask_custom)[pixel_range_start: pixel_range_end]
				data.noise = np.delete(data.oriNoise, mask_custom)[pixel_range_start: pixel_range_end]
			else:
				data.flux  = np.delete(data.oriFlux, mask_custom)
				data.wave  = np.delete(data.oriWave, mask_custom)
				data.noise = np.delete(data.oriNoise, mask_custom)
			print(len(data.wave), len(data.flux), len(data.noise))
		
		if plot_masked:
			plt.plot(data0.wave, data0.flux, 'k-', alpha=0.5, label='original data')
			plt.plot(data.wave, data.flux, 'r-', alpha=0.5, label='median combined mask data')
			plt.xlabel(r'$\lambda$ ($\AA)$')
			plt.ylabel(r'$F_{\lambda}$')
			plt.legend()
			plt.show()
			plt.close()
			#sys.exit()
		

		# continuum correction for the data
		data1    = copy.deepcopy(data)
		data     = smart.continuumTelluric(data=data, model=model)
		## constant offset correction
		const    = np.mean(data.flux) - np.mean(model.flux)
		data.flux -= const

		#print(len(data.wave),len(data.flux),len(data.noise))
		#plt.plot(data.wave, data.flux, 'r-', alpha=0.5, label='median combined mask data')
		#plt.xlabel(r'$\lambda$ ($\AA)')
		#plt.ylabel(r'$F_{\lambda}$')
		#plt.legend()
		#plt.show()
		#plt.close()
		#sys.exit()
	
		
		# defringe
		if order in defringe_list:
			data, fringe = smart.fringeTelluric(data)

		#lsf0 = smart.getLSF(data,continuum=False,test=True)
		#print("initial fitted LSF: ",lsf0)
		#model2 = smart.convolveTelluric(lsf0, data)

		# set up the path to save the calibrated spectra
		save_to_path_fits = '{}/O{}/{}'.format(save_to_path, order, data_name)

		# initial parameter txt
		file_log = open("input_params_for_cal.txt","w+")
		file_log.write("data_path {} \n".format(data_path))
		file_log.write("data_name {} \n".format(data_name))
		file_log.write("order {} \n".format(order))
		file_log.write("airmass {} \n".format(airmass))
		file_log.write("pwv {} \n".format(pwv))
		file_log.write("window_width {} \n".format(window_width))
		file_log.write("window_step {} \n".format(window_step))
		file_log.write("xcorr_range {} \n".format(xcorr_range))
		file_log.write("xcorr_step {}\n".format(xcorr_step))
		file_log.write("niter {} \n".format(niter))
		file_log.write("outlier_rej {} \n".format(outlier_rej))
		file_log.write("pixel range start {} \n".format(pixel_range_start))
		file_log.write("pixel range end {} \n".format(pixel_range_end))
		file_log.write("mask_custom {} \n".format(mask_custom))
		file_log.close()

		data_path2 = data_path + '/' + data_name + '_' + str(order) + '_all.fits'
		#print('data length:', len(data.oriWave), len(data.wave))
		time1 = time.time()

		new_wave_sol, p0, width_range_center, residual, best_shift_list = \
		smart.wavelengthSolutionFit(data, model,
			                      order=order,
			                      window_width=window_width,
			                      window_step=window_step,
			                      xcorr_range=xcorr_range,
								  xcorr_step=xcorr_step,
								  niter=niter,
								  outlier_rej=outlier_rej,
								  test=test,
								  save=save,
								  pixel_range_start=pixel_range_start,
								  pixel_range_end=pixel_range_end, 
								  save_to_path=save_to_path_fits,
								  data_path=data_path2,
								  length1 = length1,
								  apply_sigma_mask=apply_sigma_mask,
								  mask_custom=mask_custom)

		time2 = time.time()
		print("Total X correlation time: {} min".format((time2-time1)/60))

		# convert the flux back to the original data
		data       = data1
		data       = smart.continuumTelluric(data=data, model=model)
		if apply_sigma_mask:
			data.wave  = np.delete(data.wave,data.mask)
			data.flux  = np.delete(data.flux,data.mask)
			data.noise = np.delete(data.noise,data.mask)
			data.flux  = data.flux[pixel_range_start:pixel_range_end]
			data.wave  = data.wave[pixel_range_start:pixel_range_end]
			data.noise = data.noise[pixel_range_start:pixel_range_end]
		
		# plotting
		pixel       = np.delete(np.arange(length1), np.union1d(data.mask, mask_custom))
		pixel       = pixel[pixel_range_start:pixel_range_end]
		
		data.wave   = data.wave[pixel_range_start:pixel_range_end]
		data.flux   = data.flux[pixel_range_start:pixel_range_end]
		data.noise  = data.noise[pixel_range_start:pixel_range_end]
		
		linewidth   = 0.5
		stdWaveSol  = np.std(residual)
		stdWaveSolV = np.std(residual)/np.average(new_wave_sol)*299792.458
		variance2   = ((residual ** 2).sum()) / (len(residual) - 1)
		rmsWaveSol  = np.sqrt(variance2)
		rmsWaveSolV = np.sqrt(variance2)/np.average(new_wave_sol)*299792.458
		
		# add the summary to the txt file
		file_log = open("input_params_for_cal.txt","a")
		#file_log.write("*** Below is the summary *** \n")
		file_log.write("wave_sol_params {}\n".format(str(p0)))
		file_log.write("std {} Angstrom\n".format(stdWaveSol))
		file_log.write("std_vel {} km/s\n".format(stdWaveSolV))
		file_log.write("rms {} Angstrom\n".format(rmsWaveSol))
		file_log.write("rms_vel {} km/s\n".format(rmsWaveSolV))
		file_log.write("comp_time {} min\n".format(round((time2-time1)/60.,4)))
		file_log.close()

		# resampling the telluric model
		#telluric = copy.deepcopy(model)
		#telluric.flux = np.array(smart.integralResample(xh=telluric.wave, 
		#	yh=telluric.flux, xl=data.wave))
		#telluric.wave = data.wave
		# compute the LSF average broadening of the instrument (km/s)
		#vbroad = (299792458/1000)*np.mean(np.diff(telluric.wave))/np.mean(telluric.wave)
		# check the result for telluric
		#residual_telluric_data = smart.residual(data,telluric)
		
		vbroad            = (299792.458)*np.mean(np.diff(data.wave))/np.mean(data.wave)
		telluric_new      = copy.deepcopy(data)
		new_wave_sol      = smart.waveSolution(pixel,  *p0, order=order)
		telluric_new.wave = new_wave_sol
		#telluric_new.wave = smart.waveSolution(pixel,  *p0, order=order)
		telluric_new.flux  = data.oriFlux[pixel]
		telluric_new.noise = telluric_new.oriNoise[pixel]
		telluric_new       = smart.continuumTelluric(data=telluric_new, model=model)

		# get an estimate for lsf and telluric alpha
		#if apply_sigma_mask:
		lsf   = smart.getLSF(telluric_new, continuum=False)
		#else:
		#	lsf   = smart.getLSF(telluric_new)#, continuum=False)
		#lsf   = smart.getLSF2(telluric_new)#, continuum=False)
		#alpha = smart.getAlpha(telluric_new, lsf, continuum=False)
		alpha = 1.0
		print("LSF = {} km/s; alpha = {}".format(lsf, alpha))

		# make a telluric model for the input data
		telluric = smart.convolveTelluric(vbroad, data, alpha=alpha, airmass=airmass, pwv=pwv)
		residual_telluric_data = smart.residual(data,telluric)
		# add the final LSF and alpha to the txt file
		file_log = open("input_params_for_cal.txt","a")
		file_log.write("lsf {} km/s\n".format(lsf))
		file_log.write("alpha {}\n".format(alpha))
		file_log.close()
		
		telluric_new2 = smart.convolveTelluric(lsf, telluric_new, alpha=alpha, airmass=airmass, pwv=pwv)
		if order in defringe_list:
			# add back the fringe
			telluric_new2_no_fringe = copy.deepcopy(telluric_new2)
			telluric_new2.flux += fringe[pixel_range_start:pixel_range_end]

		## constant offset correction
		const    = np.mean(data.flux) - np.mean(telluric_new2.flux)
		data.flux -= const

		print("vbroad: ",lsf, " km/s")

		residual_telluric_wavesol = smart.residual(telluric_new,telluric_new2)
		
		if order == 38:
			# this is to handle a plotting bug for O38
			pass
		else:
			select = np.absolute(residual_telluric_wavesol.flux) < 5*np.std(residual_telluric_wavesol.flux)

			telluric_new_select = np.where((residual_telluric_wavesol.flux) < 5*np.std(residual_telluric_wavesol.flux))
			telluric_full       = copy.deepcopy(telluric_new2) # preserve the full wavelength range
			telluric_new.flux   = telluric_new.flux[telluric_new_select]
			telluric_new.wave   = telluric_new.wave[telluric_new_select]
			telluric_new.noise  = telluric_new.noise[telluric_new_select]
			telluric_new2.flux  = telluric_new2.flux[telluric_new_select]
			telluric_new2.wave  = telluric_new2.wave[telluric_new_select]

			residual_telluric_wavesol.flux = residual_telluric_wavesol.flux[select]
			residual_telluric_wavesol.wave = residual_telluric_wavesol.wave[select]

		# plot the telluric for comparison
		plt.tick_params(labelsize=20)
		fig = plt.figure(figsize=(16,6))
		ax1 = fig.add_subplot(111)
		ax1.plot(data.wave, data.flux, color='black', linestyle='-', 
			     label="data, STD:{}, chisquare:{}".format(\
			     round(np.nanstd(residual_telluric_data.flux),4),
			     round(smart.chisquare(data,telluric),0)),alpha=0.5,linewidth=linewidth)
		ax1.plot(telluric.wave, telluric.flux, color='red', linestyle='-',
			     label='model', alpha=0.5, linewidth=linewidth)
		ax1.plot(residual_telluric_data.wave, residual_telluric_data.flux,
			     color='blue', linestyle='-', alpha=0.5, linewidth=linewidth, label='residual')
		ax1.axhline(y=0, color='grey', linestyle=':', alpha=0.5)
		ax1.set_title("Telluric Comparison {} O{} Original Spectra".format(data_name,order),
			          y=1.15, fontsize=25)
		ax1.set_xlabel("Wavelength ($\AA$)", fontsize=20)
		ax1.set_ylabel('Normalized Flux', fontsize=20)
		ax1.tick_params(labelsize=15)
		ax1.minorticks_on()
		ax1.set_xlim(data.wave[0], data.wave[-1])
		ax1.legend(frameon=False)#loc=9, bbox_to_anchor=(0.5, -0.2),fontsize=15)

		ax2 = ax1.twiny()
		ax2.plot(pixel, data.flux, color='w', alpha=0)
		ax2.set_xlabel('Pixel',fontsize=20)
		ax2.tick_params(labelsize=15)
		ax2.set_xlim(pixel[0], pixel[-1])
		ax2.minorticks_on()

		fig.savefig("telluric_comparison_{}_O{}_1.png".format(data_name,order), 
			        bbox_inches='tight')
		plt.close()

		plt.tick_params(labelsize=20)
		fig = plt.figure(figsize=(16,8))
		gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
		ax1 = fig.add_subplot(gs[0])
		ax1.plot(telluric_new2.wave, telluric_new2.flux, 
			     color='red', linestyle='-', label='model', alpha=0.5, linewidth=linewidth)
		ax1.plot(new_wave_sol, data.flux, color='black', linestyle='-',
			     label="new wavelength solution, STD:{}, $\chi^2$:{}".format(round(np.nanstd(residual_telluric_wavesol.flux),4),
				 round(smart.chisquare(telluric_new,telluric_new2),0)),
			     alpha=0.5, linewidth=linewidth)
		if order in defringe_list:
			ax1.plot(telluric_new2_no_fringe.wave, telluric_new2_no_fringe.flux, 
				     color='blue',linestyle='-',
				     label='model w/o fringe',
				     alpha=0.5, linewidth=linewidth)
		ax1.plot(residual_telluric_wavesol.wave, residual_telluric_wavesol.flux,
			     color='blue', linestyle='-', alpha=0.5, linewidth=linewidth, label='residual')
		ax1.fill_between(new_wave_sol, -data.noise, data.noise, facecolor='grey', alpha=0.5)
		ax1.axhline(y=0, color='grey', linestyle=':', alpha=0.5)
		ax1.set_title("Telluric Comparison {} O{} Calibrated Spectra".format(data_name, order),
			          y=1.25, fontsize=25)
		ax1.set_xlabel("Wavelength ($\AA$)",fontsize=20)
		ax1.set_ylabel('Normalized Flux',fontsize=20)
		#ax1.set_ylim(-0.1, 1.1)
		ax1.set_xlim(new_wave_sol[0], new_wave_sol[-1])
		ax1.tick_params(labelsize=15)
		ax1.minorticks_on()
		ax1.legend()

		ax2 = ax1.twiny()
		#ax2.plot(new_wave_sol, data.flux, color='w', alpha=0)
		ax2.plot(data.flux, color='w', alpha=0)
		ax2.set_xlabel("Pixel",fontsize=20)
		ax2.tick_params(labelsize=15)
		#ax2.set_ylim(-0.1, 1.1)
		ax2.set_xlim(pixel[0], pixel[-1])
		ax2.minorticks_on()
		#ax2.set_xlabel('Pixel',fontsize=20)
		#ax1.legend(loc=9, bbox_to_anchor=(0.5, -0.2),fontsize=15)
		#ax2.legend(framon=False)#fontsize=15, loc='lower center', framon=False)#, bbox_to_anchor=(0.5, -0.7))

		ax3 = fig.add_subplot(gs[1])
		ax3.plot(width_range_center, residual, 'r.', alpha=0.5,
			     label="fitted wavelength function with outlier rejection, STD={} $\AA$ ={} km/s".format(\
			     np.round_(stdWaveSol,  decimals=4),
			     np.round_(stdWaveSolV, decimals=3)))
		ax3.set_ylabel("Residual ($\AA$)", fontsize=20)
		#ax3.set_ylim(-3*np.std(residual), 3*np.std(residual))
		ax3.set_xlabel('Pixel', fontsize=20)
		ax3.tick_params(labelsize=15)
		#plt.subplots_adjust(hspace=.0)
		ax3.minorticks_on()
		ax3.set_xlim(0, len(new_wave_sol))
		ax3.legend()
		plt.tight_layout()
		fig.align_labels()
		#ax3.legend(fontsize=15, loc='lower center', bbox_to_anchor=(0.5, -1.4))
		fig.savefig("telluric_comparison_{}_O{}_2.png".format(data_name,order),
			        dpi=600, bbox_inches='tight')
		plt.close()

		os.chdir(original_path)
