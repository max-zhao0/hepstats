from typing import NamedTuple
import numpy as np
from numpy.typing import ArrayLike

# from ..pytree import pt
# from ..api import Data, ParameterLike, InternalParameter, LossLike, convert_params
from .. import api

def fast_reshape(arr, shape):
    if not shape:
        return arr[0]
    elif len(shape) == 1:
        return arr
    else:
        return np.reshape(arr, shape)

class Minimum(NamedTuple):
    valid : bool
    fmin : float
    params : dict[api.ParameterKey, float | ArrayLike]

class IMinuit:
    def __init__(self, trusting=False):
        from iminuit import Minuit
        self.Minuit = Minuit
        self.trusting = trusting

    def minimize(self, 
        loss : api.LossLike, 
        params : dict[api.ParameterKey, api.ParameterLike], 
        *loss_args
    ) -> Minimum:
        param_keys = params.keys()
        param_lst = params.values()

        # Check parameters if minimizer is not trusting
        if not self.trusting:
            for p in param_lst:
                if not isinstance(p, api.ParameterLike):
                    raise ValueError("Parameters must be ParameterLike")
                try:
                    np.broadcast_shapes(np.shape(p.value), np.shape(p.upper), np.shape(p.lower), np.shape(p.floating))
                except ValueError:
                    raise ValueError("upper, lower, and floating must be broadcastable to value")

        # Flatten out everything but remembering the slices in the flattened array to put it back together
        values, uppers, lowers, floatings = [], [], [], []
        slices = np.empty(len(param_lst), dtype=slice)
        shapes = []
        
        for ip, p in enumerate(param_lst):
            vshape = np.shape(p.value)
            shapes.append(vshape)
            if vshape == tuple():
                vshape = (1,)
                
            values.append(np.atleast_1d(p.value).flatten())
            uppers.append(np.broadcast_to(p.upper, vshape).flatten())
            lowers.append(np.broadcast_to(p.lower, vshape).flatten())
            floatings.append(np.broadcast_to(p.floating, vshape).flatten())

            slices[ip] = slice(slices[ip-1].stop, slices[ip-1].stop + values[-1].size) if ip > 0 else slice(0, values[-1].size)

        p0 = np.concatenate(values)
        lowers_flat = np.concatenate(lowers)
        uppers_flat = np.concatenate(uppers)
        floatings_flat = np.concatenate(floatings)
        fixed_flat = np.invert(floatings_flat)

        # fast_reshape empirically faster for all scalar parameters
        # Explict slices are saved instead of using np.split. Empirically faster
        restore = lambda x: {key : fast_reshape(x[sl], sh) for key, sl, sh in zip(param_keys, slices, shapes)}
        call_loss = lambda x: loss(restore(x), *loss_args)
        
        minuit = self.Minuit(call_loss, p0)
        minuit.fixed[:] = fixed_flat
        minuit.limits[:] = np.stack((lowers_flat, uppers_flat), axis=-1)
        minuit.migrad()
        
        return Minimum(valid=minuit.valid, fmin=minuit.fval, params=restore(minuit.values))