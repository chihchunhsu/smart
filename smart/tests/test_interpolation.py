import pytest
import numpy as np
import smart

def test_interpolation():
	"""
	Test interpolation under various teff, logg, etc.

	Our baseline model is BT-Settl (Allard et al. 2012).

	"""

	# interpolate teff only

	model_test = smart.Model(teff=3150, logg=4.5, metal=0.0, order=str(33), modelset='btsettl08', instrument='nirspec')

	np.testing.assert_approx_equal(np.median(model_test.flux), 106404.44432097708)

	# interpolate logg only

	model_test = smart.Model(teff=2500, logg=4.85, metal=0.0, order=str(33), modelset='btsettl08', instrument='nirspec')

	np.testing.assert_approx_equal(np.median(model_test.flux), 57780.39415229681)

	# interpolate teff and logg

	model_test = smart.Model(teff=2564, logg=4.93, metal=0.0, order=str(33), modelset='btsettl08', instrument='nirspec')

	np.testing.assert_approx_equal(np.median(model_test.flux), 62229.457550836334)

if __name__ == '__main__':
	test_interpolation()