## define the telluric wavelength calibration parameters for each orders
import numpy as np

cal_param_nirspec = {
	'30':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'31':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'32':{'xcorr_range':40, 'outlier_rej':3., 	'pixel_range_start':10,	'pixel_range_end':-60 },
	'33':{'xcorr_range':25, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-50 },
	'34':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-50 },
	'35':{'xcorr_range':5, 	'outlier_rej':2., 	'pixel_range_start':10, 'pixel_range_end':-10},
	'36':{'xcorr_range':10, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-50 },
	'37':{'xcorr_range':10, 'outlier_rej':2.5, 	'pixel_range_start':50, 'pixel_range_end':-20 },
	'38':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':50, 'pixel_range_end':-20 },
	'39':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':50, 'pixel_range_end':-20 },
	'55':{'xcorr_range':5, 	'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-90 },
	'56':{'xcorr_range':5, 	'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-30 },
	'57':{'xcorr_range':20, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'58':{'xcorr_range':15, 'outlier_rej':2., 	'pixel_range_start':0, 	'pixel_range_end':-30 },
	'59':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'60':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':5, 	'pixel_range_end':-5 },
	'61':{'xcorr_range':10, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'62':{'xcorr_range':10, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'63':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'64':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'65':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':0, 	'pixel_range_end':-1 },
	'66':{'xcorr_range':15, 'outlier_rej':3., 	'pixel_range_start':10,	'pixel_range_end':-1 },
}

cal_param_igrins = {
	'77':{'xcorr_range':1.0, 'outlier_rej':2.5, 	'pixel_range_start':70, 	'pixel_range_end':-40},
}

