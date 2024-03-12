import numpy as np
import copy

def chisquare(data, model, dof=0):
    """
    Compute the chi-square value given a data and a model.
    """

    ## handling noise that has nan values
    d       = copy.deepcopy(data)
    m       = copy.deepcopy(model)
    mask    = np.isnan(d.noise)

    ## invert the boolean mask and select only the non-nan values
    d.flux  = d.flux[np.invert(mask)]
    d.noise = d.noise[np.invert(mask)]
    m.flux  = m.flux[np.invert(mask)]

    return np.nansum(( d.flux - m.flux )**2 / d.noise**2)