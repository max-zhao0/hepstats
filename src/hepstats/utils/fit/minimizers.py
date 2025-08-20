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

    # def minimize(self, 
    #     loss : api.LossLike, 
    #     params : pt.PyTree[api.ParameterLike] | api.InternalParameterCollection,  
    #     *loss_args, 
    #     data : api.Data = None
    #     ) -> Minimum:


    #     if not isinstance(params, api.InternalParameterCollection):
    #         params = api.convert_params(params)
        
    #     values_flat, unravel_func = pt.ravel(params.values)
    #     lowers_flat, _ = pt.ravel(params.lowers)
    #     uppers_flat, _ = pt.ravel(params.uppers)
    #     floatings_flat, _ = pt.ravel(params.floatings)

    #     # call_loss = lambda x: loss(unravel_func(x), *loss_args)
    #     assert len(values_flat) == len(lowers_flat) == len(uppers_flat) == len(floatings_flat)
    #     call_loss = (lambda x: loss(unravel_func(x), data, *loss_args)) if data is not None else (lambda x: loss(unravel_func(x), *loss_args))
        
    #     minuit = self.Minuit(call_loss, values_flat)
    #     minuit.fixed[:] = np.invert(floatings_flat)
    #     minuit.limits[:] = np.stack((lowers_flat, uppers_flat), axis=-1)
    #     minuit.migrad()
        
    #     return Minimum(valid=minuit.valid, fmin=minuit.fval, params=unravel_func(np.array(minuit.values)))
        
    # def minimize(self, 
    #     loss : api.LossLike, 
    #     params : pt.PyTree[api.InternalParameter | api.ParameterLike], 
    #     data : api.Data = None, 
    #     *loss_args) -> Minimum:
        
    #     param_lst, param_treedef = pt.flatten(params, is_leaf=lambda x: isinstance(x, api.ParameterLike) or isinstance(x, api.InternalParameter))
    #     if isinstance(param_lst[0], api.ParameterLike):
    #         param_lst = api.convert_params(param_lst)
    #     elif not isinstance(param_lst[0], api.InternalParameter):
    #         raise ValueError("params must contain ParameterLike or InternalParameter")

    #     values, bounds, floating, to_originals = list(zip(*param_lst)) # Iterating
    #     values_flat, value_treedef = pt.flatten(values)
    #     bounds_flat, _ = pt.flatten(list(bounds), is_leaf=lambda x: isinstance(x, tuple))
    #     fixed_flat = []
    #     for ip, f in enumerate(floating): # Iterating
    #         fixed_flat.extend([not f] * (len(values[ip]) if isinstance(values[ip], list) else 1))

    #     def restore(cvalues_flat):
    #         cvalues = pt.unflatten(value_treedef, cvalues_flat)
    #         cvalues_original = [to_originals[ip](v) for ip, v in enumerate(cvalues)] # Iterating
    #         return pt.unflatten(param_treedef, cvalues_original)

    #     assert len(values_flat) == len(bounds_flat) == len(fixed_flat)
    #     call_loss = (lambda x: loss(restore(x), data, *loss_args)) if data is not None else (lambda x: loss(restore(x), *loss_args))
        
    #     minuit = self.Minuit(call_loss, values_flat)
    #     minuit.fixed[:] = fixed_flat
    #     minuit.limits[:] = bounds_flat
    #     minuit.migrad()
        
    #     return Minimum(fmin=minuit.fval, params=restore(minuit.values), loss=loss)

<<<<<<< HEAD
    # def minimize(self, 
    #     loss : LossLike, 
    #     params : pt.PyTree[Union[InternalParameter, ParameterLike]], 
    #     data : Data = None, *loss_args
    # ) -> Minimum:
    #     param_lst, treedef = pt.flatten(params, is_leaf=lambda x: isinstance(x, ParameterLike) or isinstance(x, InternalParameter))
    #     if isinstance(param_lst[0], ParameterLike):
    #         param_lst = convert_params(param_lst)
    #     elif not isinstance(param_lst[0], InternalParameter):
    #         raise ValueError("params must contain ParameterLike or InternalParameter")
=======
    def minimize(self, 
        loss : api.LossLike, 
        params : dict[api.ParameterKey, api.ParameterLike], 
        data : api.Data = None, *loss_args
    ) -> Minimum:
        param_keys = params.keys()
        param_lst = params.values()
>>>>>>> 086c2ae (Rewrite to dict-like interface)

        # Check parameters if minimizer is not trusting
        if not self.trusting:
            for p in param_lst:
                if not isinstance(p, api.ParameterLike):
                    raise ValueError("Parameters must be ParameterLike")
                try:
                    np.broadcast_shapes(np.shape(p.value), np.shape(p.upper), np.shape(p.lower), np.shape(p.floating))
                except ValueError:
                    raise ValueError("upper, lower, and floating must be broadcastable to value")

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

            # slices[ip] = slices[ip-1] + values[-1].size if ip > 0 else values[-1].size
            slices[ip] = slice(slices[ip-1].stop, slices[ip-1].stop + values[-1].size) if ip > 0 else slice(0, values[-1].size)

        p0 = np.concatenate(values)
        lowers_flat = np.concatenate(lowers)
        uppers_flat = np.concatenate(uppers)
        floatings_flat = np.concatenate(floatings)
        fixed_flat = np.invert(floatings_flat)

        # restore = lambda x: {key : np.reshape(arr, s) for key, arr, s in zip(np.split(x, slices[:-1]), shapes, param_keys)} # explicit slices
        restore = lambda x: {key : fast_reshape(x[sl], sh) for key, sl, sh in zip(param_keys, slices, shapes)}
        call_loss = (lambda x: loss(restore(x), *loss_args)) if data is None else (lambda x: loss(restore(x), data, *loss_args))

        # assert slices[-1] == len(p0), "Slicing error in minimizer"
        
        minuit = self.Minuit(call_loss, p0)
        minuit.fixed[:] = fixed_flat
        minuit.limits[:] = np.stack((lowers_flat, uppers_flat), axis=-1)
        minuit.migrad()
        
        return Minimum(valid=minuit.valid, fmin=minuit.fval, params=restore(minuit.values))