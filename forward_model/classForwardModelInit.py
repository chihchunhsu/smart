import numpy as np
import pandas as pd

def make_spt_teff_relation(SpT):
	"""
	Use Eric Mamajek's compilation of spectral types and effective temperature relations.
	http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
	"""
	spt_teff_dic = {'M6':2850,
					'M6.5':2710,
					'M7':2650,
					'M7.5':2600,
					'M8':2500,
					'M8.5':2440,
					'M9':2400,
					'M9.5':2320,
					'L0':2250,
					'L1':2100,
					'L2':1960,
					'L3':1830,
					'L4':1700,
					'L5':1590,
					'L6':1490,
					'L7':1410,
					'L8':1350,
					'L9':1300,
					'T0':1260,
					'T1':1230,
					'T2':1200,
					'T3':1160,
					'T4':1120,
					'T4.5':1090,
					'T5':1050,
					'T5.5':1010,
					'T6':960,
					'T7':840,
					'T7.5':770,
					'T8':700,
					'T8.5':610,
					'T9':530,
					'T9.5':475,
					'Y0':420,
					'Y0.5':390,
					'Y1':350,
					'Y1.5':325,
					'Y2':250,
					}
	# separate binaries
	if '+' in SpT:
		SpT0 = SpT
		SpT = SpT.split('+')[0]
		self.binary = True
		self.SpT2 = SpT.split('+')[-1]
	elif '-' in SpT:
		SpT0 = SpT
		SpT = SpT.split('-')[0]

	if SpT in spt_teff_dic.keys():
		teff = spt_teff_dic[SpT]
	else:
		teff = int((spt_teff_dic[SpT[0:2]] + spt_teff_dic[SpT[0:1]+str(int(SpT[1:2])+1)])/2)

	return teff

class ForwardModelInit():
	"""
	Generate the MCMC priors for the forward-modeling routine.

	Examples
	--------
	>>> path = 'your/catalogue/path'
	>>> smart.ForwardModelInit.catalogue_path = path
	>>> fminit = smart.ForwardModelInit(name='J0320-0446')
	>>> fminit.makePriors()
	"""
	catalogue_path = None

	def __init__(self, name):
		self.name           = name
		self.catalogue_path = ForwardModelInit.catalogue_path
		if self.catalogue_path is not None:
			cat = pd.read_csv(self.catalogue_path)
			#self.catalogue      = cat

		self.binary         = False
			
		def make_spt_teff_relation(SpT):
			"""
			Use Eric Mamajek's compilation of spectral types and effective temperature relations.
			http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
			"""
			spt_teff_dic = {'M6':2850,
							'M6.5':2710,
							'M7':2650,
							'M7.5':2600,
							'M8':2500,
							'M8.5':2440,
							'M9':2400,
							'M9.5':2320,
							'L0':2250,
							'L1':2100,
							'L2':1960,
							'L3':1830,
							'L4':1700,
							'L5':1590,
							'L6':1490,
							'L7':1410,
							'L8':1350,
							'L9':1300,
							'T0':1260,
							'T1':1230,
							'T2':1200,
							'T3':1160,
							'T4':1120,
							'T4.5':1090,
							'T5':1050,
							'T5.5':1010,
							'T6':960,
							'T7':840,
							'T7.5':770,
							'T8':700,
							'T8.5':610,
							'T9':530,
							'T9.5':475,
							'Y0':420,
							'Y0.5':390,
							'Y1':350,
							'Y1.5':325,
							'Y2':250,
							}
			# separate binaries
			if '+' in SpT:
				SpT0 = SpT
				SpT = SpT.split('+')[0]
				self.binary = True
				self.SpT2 = SpT.split('+')[-1]
			elif '-' in SpT:
				SpT0 = SpT
				SpT = SpT.split('-')[0]

			if SpT in spt_teff_dic.keys():
				teff = spt_teff_dic[SpT]
			else:
				teff = int((spt_teff_dic[SpT[0:2]] + spt_teff_dic[SpT[0:1]+str(int(SpT[1:2])+1)])/2)

			return teff

		if self.catalogue_path is not None and self.name not in list(cat['name']):
			raise ValueError('{} is not in the catalogue.'.format(self.name))
        
		elif self.catalogue_path is not None and self.name in list(cat['name']):
			self.SpT    = cat['SpT'][cat['name'] == self.name].values[0]
			self.teff   = make_spt_teff_relation(self.SpT)
		if self.binary:
			self.teff2 = make_spt_teff_relation(self.SpT2)
		self.young  = False

	def compute_teff(self):
		self.teff   = make_spt_teff_relation(self.SpT)

	def add_secondary(self, SpT2):
		self.binary = True
		self.SpT2   = SpT2
		self.teff2 = make_spt_teff_relation(self.SpT2)
    
	def make_priors(self):
		if self.catalogue_path is not None:
			cat = pd.read_csv(self.catalogue_path)

		if self.young:
			logg_min = 4.0
			logg_max = 5.0
		else:
			logg_min = 5.0
			logg_max = 5.5

		if self.catalogue_path is not None:
			# check if RV is in the catalogue
			rv = cat['RV'][cat['name'] == self.name].values[0]
			if np.isnan(rv):
				rv_min = -100.0
				rv_max = 100.0
			else:
				rv_min = rv - 10.0
				rv_max = rv + 10.0
				            
			# check if vsini is in the catalogue
			vsini = cat['VSINI'][cat['name'] == self.name].values[0]
			if np.isnan(vsini):
				vsini_min = 0.0
				vsini_max = 70.0
			else:
				vsini_min = vsini - 5.0
				vsini_max = vsini + 5.0

		else:
			rv_min = -100.0
			rv_max = 100.0

			vsini_min = 0.0
			vsini_max = 70.0
		
		if not self.binary:	            
			self.priors =  {
							'teff_min':self.teff-200., 'teff_max':self.teff+200.,
							'logg_min':logg_min,		'logg_max':logg_max,
							'vsini_min':vsini_min,	'vsini_max':vsini_max,
							'rv_min':rv_min,		'rv_max':rv_max,
							'alpha_min':0.9,	'alpha_max':1.1,
							'A_min':-0.01,		'A_max':0.01,
							'B_min':-0.01,		'B_max':0.01,
							'N_min':0.99,		'N_max':1.01 			
							}
		else:
			flux2_scale = 0.8
			self.priors =  {
							'teff1_min':self.teff-200., 'teff1_max':self.teff+200.,
							'logg1_min':logg_min,		'logg1_max':logg_max,
							'vsini1_min':vsini_min,	'vsini1_max':vsini_max,
							'rv1_min':rv_min,		'rv1_max':rv_max,
							'teff2_min':self.teff2-200., 'teff2_max':self.teff2+200.,
							'logg2_min':logg_min,		'logg2_max':logg_max,
							'vsini2_min':vsini_min,	'vsini2_max':vsini_max,
							'rv2_min':rv_min,		'rv2_max':rv_max,
							'flux2_scale_min':flux2_scale-0.1, 'flux2_scale_max':flux2_scale+0.1,
							'airmass_min':0.9,	    'airmass_max':1.1,							
							'pwv_min':0.9,	    'pwv_max':1.1,
							#'alpha_min':0.9,	'alpha_max':1.1,
							'A_min':-0.01,		'A_max':0.01,
							'B_min':-0.01,		'B_max':0.01,
							'N_min':0.99,		'N_max':1.01 			
							}

		return self.priors
        



