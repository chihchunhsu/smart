import smart
import numpy
from scipy import special, interpolate, sparse, ndimage
import math
import copy

"""
Modification based on Jo Bovy's apogee.spec.lsf

See:
https://github.com/jobovy/apogee/blob/master/apogee/spec/lsf.py
"""

_SQRTTWO = numpy.sqrt(2.)

def _gausshermitebin(x,params,binsize):
    """Evaluate the integrated Gauss-Hermite function"""
    ncenter= params.shape[1]
    out= numpy.empty((ncenter,x.shape[1]))
    integ= numpy.empty((params.shape[0]-1,x.shape[1]))
    for ii in range(ncenter):
        poly= numpy.polynomial.HermiteE(params[1:,ii])
        # Convert to regular polynomial basis for easy integration
        poly= poly.convert(kind=numpy.polynomial.Polynomial)
        # Integrate and add up
        w1= (x[ii]-0.5*binsize)/params[0,ii]
        w2= (x[ii]+0.5*binsize)/params[0,ii]
        eexp1= numpy.exp(-0.5*w1**2.)
        eexp2= numpy.exp(-0.5*w2**2.)
        integ[0]= numpy.sqrt(numpy.pi/2.)\
            *(special.erf(w2/_SQRTTWO)-special.erf(w1/_SQRTTWO))
        out[ii]= poly.coef[0]*integ[0]
        if params.shape[0] > 1:
            integ[1]= -eexp2+eexp1
            out[ii]+= poly.coef[1]*integ[1]
        for jj in range(2,params.shape[0]-1):
            integ[jj]= (-w2**(jj-1)*eexp2+w1**(jj-1)*eexp1)\
                +(jj-1)*integ[jj-2]
            out[ii]+= poly.coef[jj]*integ[jj]
    return out

def _wingsbin(x,params,binsize,Wproftype):
    """Evaluate the wings of the LSF"""
    ncenter= params.shape[1]
    out= numpy.empty((ncenter,x.shape[1]))
    for ii in range(ncenter):
        if Wproftype == 1: # Gaussian
            w1=(x[ii]-0.5*binsize)/params[1,ii]
            w2=(x[ii]+0.5*binsize)/params[1,ii]
            out[ii]= params[0,ii]/2.*(special.erf(w2/_SQRTTWO)\
                                          -special.erf(w1/_SQRTTWO))
    return out

def unumpyack_lsf_params(lsfarr):
    """
    NAME:
       unumpyack_lsf_params
    PURPOSE:
       Unumpyack the LSF parameter array into its constituents
    InumpyUT:
       lsfarr - the parameter array
    OUTPUT:
       dictionary with unumpyacked parameters and parameter values:
          binsize: The width of a pixel in X-units
          Xoffset: An additive x-offset; used for GH parameters that vary globally
          Horder: The highest Hermite order
          Porder: Polynomial order array for global variation of each LSF parameter
          GHcoefs: Polynomial coefficients for sigma and the Horder Hermite parameters
          Wproftype: Wing profile type
          nWpar: Number of wing parameters
          WPorder: Polynomial order for the global variation of each wing parameter          
          Wcoefs: Polynomial coefficients for the wings parameters
    HISTORY:
       2015-02-15 - Written based on Nidever's code in apogeereduce - Bovy (IAS@KITP)
    """
    out= {}
    # binsize: The width of a pixel in X-units
    out['binsize']= lsfarr[0]
    # X0: An additive x-offset; used for GH parameters that vary globally
    out['Xoffset']= lsfarr[1]
    # Horder: The highest Hermite order
    out['Horder']= int(lsfarr[2])
    # Porder: Polynomial order array for global variation of each LSF parameter
    out['Porder']= lsfarr[3:out['Horder']+4]
    out['Porder']= out['Porder'].astype('int')
    nGHcoefs= numpy.sum(out['Porder']+1)
    # GHcoefs: Polynomial coefficients for sigma and the Horder Hermite parameters
    maxPorder= numpy.amax(out['Porder'])
    GHcoefs= numpy.zeros((out['Horder']+1,maxPorder+1))
    GHpar= lsfarr[out['Horder']+4:out['Horder']+4+nGHcoefs] #all coeffs
    CoeffStart= numpy.hstack((0,numpy.cumsum(out['Porder']+1)))
    for ii in range(out['Horder']+1):
        GHcoefs[ii,:out['Porder'][ii]+1]= GHpar[CoeffStart[ii]:CoeffStart[ii]+out['Porder'][ii]+1]
    out['GHcoefs']= GHcoefs
    # Wproftype: Wing profile type
    wingarr= lsfarr[3+out['Horder']+1+nGHcoefs:]
    out['Wproftype']= int(wingarr[0])
    # nWpar: Number of wing parameters
    out['nWpar']= int(wingarr[1])
    # WPorder: Polynomial order for the global variation of each wing parameter
    out['WPorder']= wingarr[2:2+out['nWpar']]
    out['WPorder']= out['WPorder'].astype('int')
    # Wcoefs: Polynomial coefficients for the wings parameters
    maxWPorder= numpy.amax(out['WPorder'])
    Wcoefs= numpy.zeros((out['nWpar'],maxWPorder+1))
    Wpar= wingarr[out['nWpar']+2:]
    WingcoeffStart= numpy.hstack((0,numpy.cumsum(out['WPorder']+1)))
    for ii in range(out['nWpar']):
        Wcoefs[ii,:out['WPorder'][ii]+1]= Wpar[WingcoeffStart[ii]:WingcoeffStart[ii]+out['WPorder'][ii]+1]
    out['Wcoefs']= Wcoefs
    return out

def raw(x,xcenter,params):
    """
    NAME:
       raw
    PURPOSE:
       Evaluate the raw APOGEE LSF (on the native pixel scale)
    InumpyUT:
       x - Array of X values for which to compute the LSF (in pixel offset relative to xcenter; the LSF is calculated at the x offsets for each xcenter if x is 1D, otherwise x has to be [nxcenter,nx]))
       xcenter - Position of the LSF center (in pixel units)
       lsfarr - the parameter array (from the LSF HDUs in the APOGEE data products)
    OUTPUT:
       LSF(x|xcenter))
    HISTORY:
       2015-02-26 - Written based on Nidever's code in apogeereduce - Bovy (IAS)
    """
    # Parse x
    if len(x.shape) == 1:
        x= numpy.tile(x,(len(xcenter),1))
    # Unumpyack the LSF parameters
    params= unumpyack_lsf_params(params)
    # Get the wing parameters at each x
    wingparams= numpy.empty((params['nWpar'],len(xcenter)))
    for ii in range(params['nWpar']):
        poly= numpy.polynomial.Polynomial(params['Wcoefs'][ii])       
        wingparams[ii]= poly(xcenter+params['Xoffset'])
    # Get the GH parameters at each x
    ghparams= numpy.empty((params['Horder']+2,len(xcenter)))
    for ii in range(params['Horder']+2):
        if ii == 1:
            # Fixed, correct for wing
            ghparams[ii]= 1.-wingparams[0]
        else:
            poly= numpy.polynomial.Polynomial(params['GHcoefs'][ii-(ii > 1)])
            ghparams[ii]= poly(xcenter+params['Xoffset'])
        # normalization
        if ii > 0: ghparams[ii]/= numpy.sqrt(2.*numpy.pi*math.factorial(ii-1))
    # Calculate the GH part of the LSF
    out= _gausshermitebin(x,ghparams,params['binsize'])
    # Calculate the Wing part of the LSF
    out+= _wingsbin(x,wingparams,params['binsize'],params['Wproftype'])
    return out

def computeAPLSF(data):
	"""
	Compute the LSF kernel for each chip
	"""
	index    = 2047
	## define lsf range and pixel centers
	xlsf     = numpy.linspace(-7.,7.,43)
	xcenter  = numpy.arange(0,4096)

	## compute LSF profiles for each chip as a function of pixel
	raw_out2_a = raw(xlsf,xcenter,data.lsfcoeff[0])
	raw_out2_b = raw(xlsf,xcenter,data.lsfcoeff[1])
	raw_out2_c = raw(xlsf,xcenter,data.lsfcoeff[2])

	## normalize
	raw_out2_a_norm = raw_out2_a/numpy.tile(numpy.sum(raw_out2_a,axis=1),(len(xlsf),1)).T
	raw_out2_b_norm = raw_out2_b/numpy.tile(numpy.sum(raw_out2_b,axis=1),(len(xlsf),1)).T
	raw_out2_c_norm = raw_out2_c/numpy.tile(numpy.sum(raw_out2_c,axis=1),(len(xlsf),1)).T

	return numpy.array([raw_out2_a_norm[index],raw_out2_b_norm[index],raw_out2_c_norm[index]])


def convolveAPLSF(model, data, lsf_array):
	"""
	Compute the LSF broadening based on apVisit header LSF coefficients.
	"""
	model2   = copy.deepcopy(model)

	## use the middle LSF for each chip
	kern_a = lsf_array[0]
	kern_b = lsf_array[1]
	kern_c = lsf_array[2]

	## select the range of model flux in each chip
	select_a = numpy.where( (model.wave <= data.oriWave0[0][0]+5) & (model.wave >= data.oriWave0[0][-1]-5) )
	select_b = numpy.where( (model.wave <= data.oriWave0[1][0]+5) & (model.wave >= data.oriWave0[1][-1]-5) )
	select_c = numpy.where( (model.wave <= data.oriWave0[2][0]+5) & (model.wave >= data.oriWave0[2][-1]-5) )

	## convolution
	convolve_flux_a_constkern = numpy.convolve(model.flux[select_a], kern_a, 'same')
	convolve_flux_b_constkern = numpy.convolve(model.flux[select_b], kern_b, 'same')
	convolve_flux_c_constkern = numpy.convolve(model.flux[select_c], kern_c, 'same')

	## combine the flux
	model2.flux = numpy.array(list(convolve_flux_c_constkern)+list(convolve_flux_b_constkern)+list(convolve_flux_a_constkern))
	model2.wave = numpy.array(list(model.wave[select_c])+list(model.wave[select_b])+list(model.wave[select_a]))

	return model2









	