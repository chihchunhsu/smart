import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5

def compute_uvw_velocity(ra_J2000, dec_J2000, parallax, rv, mu_ra, mu_dec, e_parallax, e_rv, e_mu_ra, e_mu_dec, correct_lsr=True):
	"""
	Compute the Galactic UVW space velocity based on the formulation in Johnson and Soderblom (1987).

	Parameters
	----------
	ra 			:	float
					RA of the source in degrees
	dec 		:	float
					Dec of the source in degrees

	parallax	:	float
					the parallax in mas
	rv			:	float
					the radial velocity in km/s
	mu_ra		:	float
					the proper motion in right ascension in mas/yr
	mu_dec		:	float
					the proper motion in declination in mas/yr

	e_parallax	:	float
					the error of parallax in mas
	e_rv		:	float
					the error of radial velocity in km/s
	e_mu_ra		:	float
					the error of proper motion in right ascension in mas/yr
	e_mu_dec	:	float
					the error of proper motion in declination in mas/yr

	Optional Parameters
	-------------------
	correct_lsr	:	bool
					If True: uvw corrected to the LSR

	Returns
	-------
	uvw 		:	array-like
					UVW velocities in km/s
	e_uvw 		:	array-like
					errors of UVW velocities in km/s

	"""
	## convert proper motions and parallax from mas to arcsec
	parallax   /= 1000
	mu_ra 	   /= 1000
	mu_dec 	   /= 1000

	e_parallax /= 1000
	e_mu_ra    /= 1000
	e_mu_dec   /= 1000

	## convert ra and dec into radians (the paper uses equinox 1950)
	coord_J2000 = SkyCoord(ra_J2000*u.deg, dec_J2000*u.deg, unit='deg', frame='icrs')

	coord_J1950 = coord_J2000.transform_to(FK5(equinox='J1950.0'))

	ra          = coord_J1950.ra.value
	dec         = coord_J1950.dec.value

	## degree to radian conversion
	deg_to_rad  = np.pi/180

	## define the A matrix
	A_ra      = np.array([	[	+np.cos(ra*deg_to_rad),		+np.sin(ra*deg_to_rad),	0],
							[	+np.sin(ra*deg_to_rad),		-np.cos(ra*deg_to_rad),	0],
							[						 0,							0, -1]])

	A_dec	  = np.array([	[	+np.cos(dec*deg_to_rad),	 0,	-np.sin(dec*deg_to_rad)],
							[						  0, 	-1,						  0],
							[	-np.sin(dec*deg_to_rad),	 0,	-np.cos(dec*deg_to_rad)]])

	A         = A_ra.dot(A_dec)

	#A0 		= np.array([[ 	+np.cos(ra*deg_to_rad)*np.cos(dec*deg_to_rad), -np.sin(ra*deg_to_rad), -np.cos(ra*deg_to_rad)*np.sin(dec*deg_to_rad)],
	#					[	+np.sin(ra*deg_to_rad)*np.cos(dec*deg_to_rad), +np.cos(ra*deg_to_rad), -np.sin(ra*deg_to_rad)*np.sin(dec*deg_to_rad)],
	#					[	+np.sin(dec*deg_to_rad) 					 , 				   	    0, +np.cos(dec*deg_to_rad)					   ]])

	## define RA and Dec for the North Galactic Pole (NGP) in degrees
	ra_ngp  = 192.25
	dec_ngp = 27.4
	theta0  = 123 # the position angle of NGP relative to great semi-circle of the North Celetial Pole and the zero Galactic longitude
	
	T1      = np.array([[  +np.cos(theta0*deg_to_rad), +np.sin(theta0*deg_to_rad),  0],
						[  +np.sin(theta0*deg_to_rad), -np.cos(theta0*deg_to_rad),  0],
						[  						    0,			 			 	0,  +1]])

	T2      = np.array([[-np.sin(dec_ngp*deg_to_rad),  0, +np.cos(dec_ngp*deg_to_rad)],
						[						   0, -1, 						  	0],
						[+np.cos(dec_ngp*deg_to_rad),  0, +np.sin(dec_ngp*deg_to_rad)]])

	T3      = np.array([[  +np.cos(ra_ngp*deg_to_rad), +np.sin(ra_ngp*deg_to_rad),  0],
						[  +np.sin(ra_ngp*deg_to_rad), -np.cos(ra_ngp*deg_to_rad),  0],
						[						    0,						    0, +1]])

	## define the T matrix
	T       = T1.dot(T2.dot(T3))

	## B matrix = TA
	B       = T.dot(A)

	## uvw matrix
	k       = 1.4959787 * 10**8 / 365.24219879 / 24 /3600 #4.74057 # AU/tropical yr (km/s)
	uvw     = B.dot(np.array([	[rv], 
								[k * mu_ra 	/ parallax], 
								[k * mu_dec / parallax]]))

	## solar uvw from Schonrich et al. (2010)
	uvw_solar = np.array([	[11.1],	[12.24], [7.25]	])

	C       = B**2
	e_uvw2  = C.dot(np.array([	[ e_rv**2], 
								[ (k/parallax)**2 * ( e_mu_ra**2  + ( mu_ra  * e_parallax / parallax )**2 )], 
								[ (k/parallax)**2 * ( e_mu_dec**2 + ( mu_dec * e_parallax / parallax )**2 )]	])) \
					+ 2 * mu_ra * mu_dec * k**2 * e_parallax**2 / parallax**4 * \
					np.array([ 	[ B[0][1]*B[0][2] ], 
								[ B[1][1]*B[1][2] ], 
								[ B[2][1]*B[2][2] ] ])

	if correct_lsr: uvw += uvw_solar

	return uvw, np.sqrt(e_uvw2)
	