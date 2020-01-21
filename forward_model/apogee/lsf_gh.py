import numpy as np

#####################################################################################
# Python implementation of gausshermite.pro and lsf_gh.pro							#
# Reference:																		#
# (1) https://github.com/callendeprieto/ferre/blob/master/src/lsf_gh.f90			#
# (2) https://github.com/callendeprieto/ferre/blob/master/src/gausshermite.f90		#
#####################################################################################

def gausshermite(x,n,par,npar,ghfun):
	nherm =5
	pi    = np.pi
	sqr2  = np.sqrt(2.)
	sqr3  = np.sqrt(3.)
	sqr24 = np.sqrt(24.)
	sqrpi = np.sqrt(pi)

