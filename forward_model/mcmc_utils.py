

def contruct_path():
	"""
	Construct paths for science, telluric data, and saving MCMC outputs.
	"""

	return data_path, tell_path, save_to_path

def generate_initial_priors_and_limits(sp_type):
	priors =  { 'teff_min':max(sp_type_temp_dic[sp_type]+200,500),  'teff_max':min(sp_type_temp_dic[sp_type]-200,3500),
				'logg_min':3.5,                            'logg_max':5.0,
				'vsini_min':0.0,                           'vsini_max':100.0,
				'rv_min':-200.0,                           'rv_max':200.0,
				'alpha_min':0.9,                           'alpha_max':1.1,
				'A_min':-0.01,                             'A_max':0.01,
				'B_min':-0.01,                             'B_max':0.01,
				'N_min':0.99,                              'N_max':1.01 			}
		
	limits =  { 'teff_min':max(sp_type_temp_dic[sp_type]-200,500),  'teff_max':min(sp_type_temp_dic[sp_type]+200,3500),
				'logg_min':3.5,                            'logg_max':5.0,
				'vsini_min':0.0,                           'vsini_max':100.0,
				'rv_min':-200.0,                           'rv_max':200.0,
				'alpha_min':0.1,                           'alpha_max':2.0,
				'A_min':-1.0,                              'A_max':1.0,
				'B_min':-0.6,                              'B_max':0.6,
				'N_min':0.10,                              'N_max':2.50 				}
	return priors, limits

def generate_final_priors_and_limits(barycorr, save_to_path1):
	df = pd.read_csv(save_to_path1+'/mcmc_result.txt', sep=' ', header=None)
	#barycorr = smart.barycorr(data.header).value
	priors1 =  {'teff_min':max(float(df[1][0])-20,900),   'teff_max':min(float(df[1][0])+20,1300),
				'logg_min':max(float(df[1][1])-0.001,3.5), 'logg_max':min(float(df[1][1])+0.001,5.5),
				'vsini_min':float(df[1][2])-0.1,           'vsini_max':float(df[1][2])+0.1,
				'rv_min':float(df[1][3])-0.1-barycorr,     'rv_max':float(df[1][3])+0.1-barycorr,
				'alpha_min':float(df[1][4])-0.01,          'alpha_max':float(df[1][4])+0.01,
				'A_min':float(df[1][5])-0.001,             'A_max':float(df[1][5])+0.001,
				'B_min':float(df[1][6])-0.001,             'B_max':float(df[1][6])+0.001,
				'N_min':float(df[1][7])-0.001,             'N_max':float(df[1][7])+0.001  		}
		
	limits1 =  { #'teff_min':max(sp_type_temp_dic[sp_type]+200,500),  'teff_max':min(sp_type_temp_dic[sp_type]-200,3500),
				'teff_min':max(float(df[1][0])-200,500),  'teff_max':min(float(df[1][0])+200,3500),
				'logg_min':3.5,                            'logg_max':5.0,
				'vsini_min':0.0,                           'vsini_max':100.0,
				'rv_min':-200.0,                           'rv_max':200.0,
				'alpha_min':0.1,                           'alpha_max':2.0,
				'A_min':-1.0,                              'A_max':1.0,
				'B_min':-0.6,                              'B_max':0.6,
				'N_min':0.10,                              'N_max':2.50 				}
	return priors1, limits1

def mask_from_initial_mcmc(custom_mask, save_to_path):
	save_to_path1  = save_to_path + '/init_mcmc'
	save_to_path2  = save_to_path + '/final_mcmc'

	df = pd.read_csv(save_to_path1+'/mcmc_result.txt', sep=' ', header=None)

	mcmc_dic = {'teff':float(df[1][0]),
				'logg':float(df[1][1]),
				'vsini':float(df[1][2]),
				'rv':float(df[1][3]),
				'alpha':float(df[1][4]),
				'A':float(df[1][5]),
				'B':float(df[1][6]),
				'N':float(df[1][7]),
				'lsf':lsf
				}

	data2 = copy.deepcopy(data)
	data2.wave = data2.wave[pixel_start: pixel_end]
	data2.flux = data2.flux[pixel_start: pixel_end]

	model = smart.makeModel(mcmc_dic['teff'], mcmc_dic['logg'],0,
		mcmc_dic['vsini'], mcmc_dic['rv']-barycorr, mcmc_dic['alpha'], 
		mcmc_dic['B'], mcmc_dic['A'], lsf=mcmc_dic['lsf'], data=data, order=data.order)
	pixel = np.delete(np.arange(len(data2.oriWave)),data2.mask)[pixel_start: pixel_end]
	custom_mask2 = pixel[np.where(np.abs(data2.flux-model.flux[pixel_start: pixel_end]) > outlier_rejection*np.std(data2.flux-model.flux[pixel_start: pixel_end]))]
		
	plt.figure(figsize=(16,6))
	plt.plot(np.arange(1024),data.flux,'k-',alpha=0.5)
	plt.plot(np.arange(1024),model.flux,'r-',alpha=0.5)
	plt.plot(pixel[np.where(np.abs(data2.flux-model.flux[pixel_start: pixel_end]) < outlier_rejection*np.std(data2.flux-model.flux[pixel_start: pixel_end]))],
		data2.flux[np.where(np.abs(data2.flux-model.flux[pixel_start: pixel_end]) < outlier_rejection*np.std(data2.flux-model.flux[pixel_start: pixel_end]))],'b-',alpha=0.5)
	plt.ylabel('cnt/s')
	#plt.xlabel('wavelength ($\AA$)')
	plt.xlabel('pixel')
	plt.minorticks_on()
	if not os.path.exists(save_to_path2):
		os.makedirs(save_to_path2)
	plt.savefig(save_to_path2+'/spectrum_mask.png')
	#plt.show()
	plt.close()

	custom_mask2 = np.append(custom_mask2,np.array(custom_mask))
	custom_mask2.sort()

	return custom_mask2

#def coadd_spectra():

#def save_to_rv_catalogue():


#def log_file():

## The relation is from http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
sp_type_temp_dic = {
'M0':3870, 'M0.5':3800, 'M1':3700, 'M1.5':3650, 'M2':3550, 'M2.5':3500, 'M3':3410, 'M3.5':3250, 'M4':3200, 'M4.5':3100,
'M5':3030, 'M5.5':3000, 'M6':2850, 'M6.5':2710, 'M7':2650, 'M7.5':2600, 'M8':2500, 'M8.5':2440, 'M9':2400, 'M9.5':2320,
'L0':2250, 'L1':2100, 'L2':1960, 'L3':1830, 'L4':1700, 'L5':1590, 'L6':1490, 'L7':1410, 'L8':1350, 'L9':1300,
'T0':1260, 'T1':1230, 'T2':1200, 'T3':1160, 'T4':1120, 'T4.5':1090, 'T5':1050, 'T5.5':1010, 'T6':960, 'T7':840, 'T8':700
}
