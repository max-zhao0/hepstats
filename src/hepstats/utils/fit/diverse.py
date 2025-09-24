from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping, Collection
import warnings
# from contextlib import ExitStack, contextmanager, suppress

import numpy as np
from .. import api
from .poi import POI
from .minimizers import Minimum, IMinuit
from collections import OrderedDict

def get_ndims(dataset):
    """Return the number of dimensions in the dataset"""
    return len(dataset.obs)

def sample(model : api.UnbinnedModelLike, params : dict[api.ParameterKey, float | ArrayLike], nsamples : int, minimizer : api.MinimizerLike):
    call_pdf = lambda x: model.pdf(x, params)
    if not model.pdf_vectorized:
        call_pdf = np.vectorize(call_pdf)

    lower = np.atleast_1d(model.lower)
    upper = np.atleast_1d(model.upper)

    # Find maximum the pdf reaches over the domain
    TempParam = namedtuple("TempParam", ["value", "upper", "lower", "floating"])
    xparam = TempParam(np.mean([lower, upper], axis=0), upper, lower, True)
    minimum = minimizer.minimize(lambda x: -call_pdf(x), {"x" : xparam})
    max_height = - minimum.fmin

    # Number of throws such that the expected number of passing samples is nsamples.
    def run_mc(nthrows):
        nthrows = int(nthrows)
        x = np.random.uniform(lower, upper, size=(nthrows, len(lower)))
        y = np.random.uniform(0, max_height, size=nthrows)
        return x[call_pdf(x) > y]
    
    expected_nthrows = np.prod(upper - lower) * nsamples * max_height
    res = run_mc(expected_nthrows * 1.2)
    while len(res) < nsamples:
        res = np.concatenate([res, run_mc(expected_nthrows * 0.2)], axis=0)
    res = res[:nsamples]
    
    return res if np.ndim(model.lower) > 0 else res[:,0]

def pll(pois : POI | Collection[POI],
    minimizer : api.MinimizerLike, 
    loss : api.LossLike, 
    params : dict[api.ParameterKey, api.InternalParameter],
    *loss_args,
    ntrials_fit : int = 1,
    init=None
) -> Minimum:
    """Compute minimum profile likelihood for fixed given parameters values."""
    del init  # unused currently
    if isinstance(pois, POI):
        pois = [pois]

    fitting_params = {k : api.InternalParameter(params[k], trusting=True) for k in params}
    for p in pois:
        fitting_params[p.param_key].value = p.value
        fitting_params[p.param_key].floating = False

    if np.any([np.any(fitting_params[k].floating) for k in fitting_params]):
        for _ in range(ntrials_fit):
            minimum = minimizer.minimize(loss, fitting_params, *loss_args)
            if minimum.valid:
                return minimum
            else:
                for k in fitting_params:
                    if fitting_params[k].floating:
                        fitting_params[k].value += np.random.normal(0, 0.02)

        msg = "No valid minimum was found when fitting the loss function for the alternative"
        msg += f"hypothesis ({pois}), after {ntrials_fit} trials."
        warnings.warn(msg, stacklevel=2)

        nominal_values = {k : params[k].value for k in params}
        return Minimum(fmin=loss(nominal_values, *loss_args), valid=False, params=nominal_values)
    else:
        nominal_values = {k : params[k].value for k in params}
        return Minimum(fmin=loss(nominal_values, *loss_args), valid=True, params=nominal_values)

def get_nevents(dataset):
    """Returns the number of events in the dataset"""

    return get_value(dataset.nevents)