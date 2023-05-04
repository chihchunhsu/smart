import numpy as np

## The code is from apogee_tools

def lsf_rotate(deltaV, Vsini, epsilon = False, velgrid = False):
	"""Create a 1-d convolution kernel to broaden a spectrum from a rotating star

	Can be used to derive the broadening effect (line spread function; LSF) 
	due to  rotation on a synthetic stellar spectrum. Assumes constant 
	limb darkening across the disk.

	Adapted from rotin3.f in the SYNSPEC software of Hubeny & Lanz 
	.http://nova.astro.umd.edu/index.html	Also see Eq. 17.12 in 
	"The Observation and Analysis of Stellar Photospheres" by D. Gray (1992)

	Written,		   W. Landsman		  November 2001
	Ported to Python,  C. Theissen		  June 2017

	Args:
		deltaV (float):   Numeric scalar giving the step increment (in km/s) in the output 
			rotation kernel. 

		Vsini (float):	The rotational velocity projected  along the line of sight (km/s)

		epislon (bool; optional): Numeric scalar giving the limb-darkening coefficient, 
			default = 0.6 which is typical for  photospheric lines. The
			specific intensity I at any angle theta from the specific intensity
			Icen at the center of the disk is given by:
	 
			Example: I = Icen*(1-epsilon*(1-cos(theta))

		velgrid (bool; optional): Returns a vector with the same number of elements as LSF


	Returns:
		LSF: The convolution kernel vector for the specified rotational velocity.
			The  number of points in LSF will be always be odd (the kernel is
			symmetric) and equal to  either ceil(2*Vsini/deltav) or 
			ceil(2*Vsini/deltav) +1 (whichever number is odd). LSF will 
			always be of type float.
	
			To actually compute the broadening. the spectrum should be convolved
			with the rotational LSF. 

		velgrid (optional): if keyword 'velgrid' is True, returns a vector with
			the same number of elements as LSF.

	"""

	"""
	if N_params() LT 1 then begin
		 print 'Syntax - rkernel = lsf_rotate(deltav, vsini)'
		 print '	  Input Keyword: Epsilon'
		 print '	  Output Keyword: Velgrid'
		 return,-1
	endif
	"""

	if epsilon == False: 
		epsilon = 0.6
	e1 = 2. * (1. - epsilon)
	e2 = np.pi * epsilon/2.
	e3 = np.pi * (1. - epsilon/3.)

	npts = np.ceil(2.*Vsini/deltaV)

	if npts % 2 == 0: 
		npts = npts +1

	nwid = npts // 2.

	x	= (np.arange(npts) - nwid)

	x	= x * deltaV / Vsini 

	x1 = abs(1.0 - x**2)

	if velgrid: 
		velgrid = x * Vsini
		return (e1*np.sqrt(x1) + e2*x1)/e3, velgrid

	else:
		return (e1*np.sqrt(x1) + e2*x1)/e3
   
	  


def broaden(wave, flux, vbroad, rotate=False, gaussian=True):
	"""Broaden a spectrum with either a rotational or Gassuain kernel

	Written,		   A. Burgasser		 Deslember 20XX
	Ported to Python,  C. Theissen		  June 2017

	Args:
		wave (float array):   Array with the wavelengths. 

		flux (float array):   Array with the flux values corredponsing to the wavelengths.

		vbroad (float): Broadening value in km/s

		rotate (bool; optional): Compute the rotational broadening kernel

		gaussian (bool; optional): Compute the Gaussian broadening kernel


	Returns:
		The spectrum convolved with the broadening kernel.
		This will have the same length as the input flux array.

	"""

	cvel = 299792.458

	vres = cvel*np.median( abs((wave - np.roll(wave,1)) / wave))

	if rotate: 
		kern = lsf_rotate(vres, vbroad)
 
	elif gaussian: 
		if np.ceil(20.*vbroad/vres) % 2 == 0:
			x	= np.arange(np.ceil(20.*vbroad/vres)+1)
		else:
			x	= np.arange(np.ceil(20.*vbroad/vres))

		x	= (x / np.max(x)-0.5)*20.

		kern = np.exp(-0.5*x**2)
 
	else:
		if np.ceil(20.*vbroad/vres) % 2 == 0:
			x	= np.arange(np.ceil(20.*vbroad/vres)+1)*10.
		else:
			x	= np.arange(np.ceil(20.*vbroad/vres))*10.

		kern = np.exp(-0.5*x**2)

	kern	 = kern/np.sum(kern)

	return np.convolve(flux, kern, 'same')

def rot_int_cmj(wave, flux, vsini, epsilon=0.6, nr=10, ntheta=100, dif = 0.0):
	"""

	Adapted from Carvalho & Johns-Krull (2023).
	See https://github.com/Adolfo1519/RotBroadInt

	A routine to quickly rotationally broaden a spectrum in linear time.
	INPUTS:
	s - input spectrum
	w - wavelength scale of the input spectrum
	vsini (km/s) - projected rotational velocity

	OUTPUT:
	ns - a rotationally broadened spectrum on the wavelength scale w
	OPTIONAL INPUTS:
	eps (default = 0.6) - the coefficient of the limb darkening law
	nr (default = 10) - the number of radial bins on the projected disk

	ntheta (default = 100) - the number of azimuthal bins in the largest radial annulus
	note: the number of bins at each r is int(r*ntheta) where r < 1
	
	dif (default = 0) - the differential rotation coefficient, applied according to the law
	Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)). Dif = .675 nicely reproduces the law 
	proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS. Dif = .23 is 
	similar to observed solar differential rotation. Note: the th in the above expression is 
	the stellar co-latitude, not the same as the integration variable used below. This is a 
	disk integration routine.
	"""

	ns = np.copy(flux)*0.0
	tarea = 0.0
	dr = 1./nr
	for j in range(0, nr):
		r = dr/2.0 + j*dr
		area = ((r + dr/2.0)**2 - (r - dr/2.0)**2)/int(ntheta*r) * (1.0 - epsilon + epsilon*np.cos(np.arcsin(r)))
		for k in range(0,int(ntheta*r)):
			th = np.pi/int(ntheta*r) + k * 2.0*np.pi/int(ntheta*r)
			if dif != 0:
				vl = vsini * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0*np.cos(2.0*np.arccos(r*np.cos(th))))
				ns += area * np.interp(wave + wave*vl/2.99792458e5, wave, flux)
				tarea += area
			else:
				vl = r * vsini * np.sin(th)
				ns += area * np.interp(wave + wave*vl/2.99792458e5, wave, flux)
				tarea += area
		  
	return ns/tarea
	