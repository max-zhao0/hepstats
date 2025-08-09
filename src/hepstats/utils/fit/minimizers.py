from typing import NamedTuple
import numpy as np

from ..pytree import pt
# from ..api import Data, ParameterLike, InternalParameter, LossLike, convert_params
from .. import api

class Minimum(NamedTuple):
    valid : bool
    fmin : float
    params : pt.PyTree[np.ndarray[float]]

class IMinuit:
    def __init__(self):
        from iminuit import Minuit
        self.Minuit = Minuit

    def minimize(self, 
        loss : api.LossLike, 
        params : pt.PyTree[api.ParameterLike] | api.InternalParameterCollection,  
        *loss_args, 
        data : api.Data = None
        ) -> Minimum:


        if not isinstance(params, api.InternalParameterCollection):
            params = api.convert_params(params)
        
        values_flat, unravel_func = pt.ravel(params.values)
        lowers_flat, _ = pt.ravel(params.lowers)
        uppers_flat, _ = pt.ravel(params.uppers)
        floatings_flat, _ = pt.ravel(params.floatings)

        # call_loss = lambda x: loss(unravel_func(x), *loss_args)
        assert len(values_flat) == len(lowers_flat) == len(uppers_flat) == len(floatings_flat)
        call_loss = (lambda x: loss(unravel_func(x), data, *loss_args)) if data is not None else (lambda x: loss(unravel_func(x), *loss_args))
        
        minuit = self.Minuit(call_loss, values_flat)
        minuit.fixed[:] = np.invert(floatings_flat)
        minuit.limits[:] = np.stack((lowers_flat, uppers_flat), axis=-1)
        minuit.migrad()
        
        return Minimum(valid=minuit.valid, fmin=minuit.fval, params=unravel_func(np.array(minuit.values)))
        
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

    #     values, bounds, floating = list(zip(*param_lst))
    #     fixed = np.invert(floating)

    #     slices = np.empty(len(values))
    #     values_arrays = []
    #     bounds_arrays = []
    #     for ip, (v, b) in enumerate(zip(values, bounds)):
    #         values_arrays.append(np.atleast_1d(v))
    #         bounds_arrays.append(np.atleast_2d(b))
    #         slices[ip] = slices[ip-1] + values_arrays[-1].size if ip > 0 else values_arrays[-1].size

    #     assert slices[-1] == len(values_arrays), "Slicing error in minimizer"

    #     split = lambda x: [arr[0] if arr.size == 1 else arr for arr in np.split(x, slices[:-1])]
        
    #     if np.max(values_sizes) > 1:
    #         p0 = np.concatenate(values_arrays)
    #         flat_bounds = np.concatenate(bounds_arrays, axis=0)
    #         call_loss = lambda x: loss(pt.unflatten(treedef, split(x)))
    #     else:
    #         p0 = values
    #         flat_bounds = bounds
    #         call_loss = lambda x: loss(pt.unflatten(treedef, x))
        
    #     minuit = self.Minuit(..., p0, fixed=..., limits=flat_bounds)
    #     minuit.migrad()
    #     return Minimum(minuit.fval, pt.unflatten(treedef, split(minuit.values)))