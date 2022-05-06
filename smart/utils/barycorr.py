#!/usr/bin/env python
#
# History
# Feb. 01 2018 Dino Hsu
# Feb. 03 2019 Chris Theissen
# Feb. 05 2019 Dino Hsu
# The barycentric correction function using Astropy
#

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.utils.iers import conf
conf.auto_max_age = None

# Keck information
# I refer to the telescope information from NASA
# https://idlastro.gsfc.nasa.gov/ftp/pro/astro/observatory.pro
# (longitude - observatory longitude in degrees *west*)
# Need to convert the longitude to the definition in pyasl.helcorr
# obs_long: Longitude of observatory (degrees, **eastern** direction is positive)

# Mcdonald Observatory
# https://cddis.nasa.gov/site_cat/mcdo.html
# Latitude: N 30 40' 48.94384"
# Longitude: W 104 0' 54.67839"
# Elevation: 2025.960 meters


#`~astropy.coordinates.Longitude` or float
#   Earth East longitude.  Can be anything that initialises an
#   `~astropy.coordinates.Angle` object (if float, in degrees)

def barycorr(header, instrument='nirspec'):
	"""
	Calculate the barycentric correction using Astropy.
	
	Input:
	header (fits header): using the keywords UT, RA, and DEC

	Output:
	barycentric correction (float*u(km/s))

	"""
	if (instrument == 'nirspec') or (instrument == 'hires'):
		longitude = 360 - (155 + 28.7/60 ) # degrees
		latitude  = 19 + 49.7/60 #degrees
		altitude  = 4160.

		keck = EarthLocation.from_geodetic(lat=latitude*u.deg, lon=longitude*u.deg, height=altitude*u.m)

		date    = Time(header['DATE-OBS'], scale='utc')
		jd      = date.jd

		if (jd >= 2458401.500000) or (instrument=='hires'): # upgraded NIRSPEC or HIRES
			ut  = header['DATE-OBS'] + 'T' + header['UT'] 
			ra  = header['RA']
			dec = header['DEC']
			sc  = SkyCoord('%s %s'%(ra, dec), unit=(u.hourangle, u.deg), equinox='J2000', frame='fk5')
		else:
			ut  = header['DATE-OBS'] + 'T' + header['UTC']
			ra  = header['RA']
			dec = header['DEC']
			sc  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, equinox='J2000', frame='fk5')

		barycorr = sc.radial_velocity_correction(obstime=Time(ut, scale='utc'), location=keck)

	elif instrument == 'apogee':
		longitude = 360 - (105 + 49.2/60 ) # degrees
		latitude  =  32 + 46.8/60 #degrees
		altitude  = 2798.

		apogee = EarthLocation.from_geodetic(lat=latitude*u.deg, lon=longitude*u.deg, height=altitude*u.m)

		ut      = header['UT-MID']
		ra      = header['RA']
		dec     = header['DEC']
		sc      = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, equinox='J2000', frame='fk5')

		#apogee  = EarthLocation.of_site('Apache Point Observatory')

		barycorr = sc.radial_velocity_correction(obstime=Time(ut, scale='utc'), location=apogee)

	elif instrument == 'igrins':
		# McDonalds Observatory
		longitude = 360 - (104+0/60+54.67839/3600) # degrees
		latitude  =  +1*(30+40/60+48.94384/3600) #degrees
		altitude  = 2025.960

		apogee = EarthLocation.from_geodetic(lat=latitude*u.deg, lon=longitude*u.deg, height=altitude*u.m)

		ut      = header['DATE-OBS']
		tmp_ra  = header['TELRA'].split('+')[1].split(':')
		ra      = 15* ( float(tmp_ra[0]) + float(tmp_ra[1])/60 + float(tmp_ra[2])/3600 )
		if '-' in header['TELDEC']:
			tmp_dec = header['TELDEC'].split('-')[1].split(':')
			dec      = -1 * (float(tmp_dec[0]) + float(tmp_dec[1])/60 + float(tmp_dec[2])/3600)
		else:
			tmp_dec = header['TELDEC'].split('+')[1].split(':')
			dec      = +1 * (float(tmp_dec[0]) + float(tmp_dec[1])/60 + float(tmp_dec[2])/3600)
		sc      = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, equinox='J2000', frame='fk5')

		#apogee  = EarthLocation.of_site('Apache Point Observatory')

		barycorr = sc.radial_velocity_correction(obstime=Time(ut, scale='utc'), location=apogee)

	
	return barycorr.to(u.km/u.s)
