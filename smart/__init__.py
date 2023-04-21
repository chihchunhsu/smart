__version__ = "0.0.3"

from .initialize import *
from .utils.barycorr import barycorr
from .forward_model.classModel import Model
from .forward_model.classSpectrum import Spectrum
from .forward_model.classForwardModelInit import ForwardModelInit
from .forward_model.integralResample import integralResample
from .forward_model.InterpolateModel import InterpModel
from .forward_model.rotation_broaden import lsf_rotate, broaden
from .forward_model.continuum import *
from .forward_model.tellurics import InterpTelluricModel, convolveTelluric
from .forward_model.model_fit import *
from .forward_model.fringe_model import double_sine, doub_sine_wave_dependent, double_sine_fringe, double_sine_fringe_telluric
from .forward_model.mcmc import run_mcmc, telluric_mcmc, run_mcmc2, run_mcmc3
from .forward_model.apogee.lsf_function import computeAPLSF, convolveAPLSF
try:
	from .forward_model.apogee.lsf import eval
except FileNotFoundError:
	print("You need to put the APOGEE LSF and wavelength fits files under forward_model/apogee. See README.md for more details.")
	pass
from .forward_model.apogee.bitmask import bits_set
from .wavelength_calibration.telluric_wavelength_fit import *
from .wavelength_calibration.residual import residual
from .utils.stats import chisquare
from .utils.addKeyword import add_nsdrp_keyword, add_wfits_keyword
from .utils.listTarget import makeTargetList
from .utils.compute_uvw_velocity import compute_uvw_velocity
from .utils.compute_galactic_thin_thick_disk import compute_galactic_pop_prob
from .utils.interpolations import bilinear_interpolation, trilinear_interpolation, quadlinear_interpolation
try:
	from .utils.defringeflat import defringeflat, defringeflatAll
except ImportError:
	print("You need to install the wavelets package (https://github.com/aaren/wavelets) to use defringeflat functions.")
	pass
#from .utils.subtractDark import subtractDark
