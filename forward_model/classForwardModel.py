import tellurics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import emcee
import corner
from multiprocessing import Pool
import nirspec_fmp as nsp
import sys
import time
import copy

class ForwardModel():
	def __init__(self, data, parameters, priors):
		self.parameters = parameters
		self.priors     = priors
		self.data       = data

	def run_mcmc(self):
		pass

	def generate_mask(self):
		pass

	def generate_output(self):
		pass

class ForwardModelTelluric():
	def __init__(self, data, parameters, priors=None):
		self.data       = data
		self.parameters = parameters
		self.priors     = {	'lsf_min':2.0  		,	'lsf_max':10.0,
							'airmass_min':1.0   ,	'airmass_max':1.5,
							'pwv_min':0.50 		,	'pwv_max':2.50,
							'A_min':-500.0 		,	'A_max':500.0,
							'B_min':-0.04  	    ,	'B_max':0.04    }
		

	def run_mcmc(self):
		pass

	def generate_mask(self):
		pass

	def generate_output(self):
		pass

