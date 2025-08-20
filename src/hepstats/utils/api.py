"""
Module for testing a fitting library validity with hepstats.

A fitting library should provide six basic objects:

    * model / probability density function
    * parameters of the models
    * data
    * loss / likelihood function
    * minimizer
    * fitresult (optional)

A function for each object is defined in this module, all should return `True` to work
with hepstats.

The `zfit` API is currently the standard fitting API in hepstats.

"""

# from __future__ import annotations

# import warnings

# import uhi.typing.plottable

from typing import runtime_checkable, Protocol, TypeVar, Callable, Union, NamedTuple, Hashable
from numpy.typing import ArrayLike
import numpy as np
import warnings

ParameterKey = TypeVar("ParameterKey", bound=Hashable)
DataKey = TypeVar("DataKey", bound=Hashable)
Data = TypeVar("Data", bound=dict[DataKey, ArrayLike])

@runtime_checkable
class ParameterLike(Protocol):
    value : float | ArrayLike
    upper : float | ArrayLike
    lower : float | ArrayLike
    floating : bool

@runtime_checkable
class LossLike(Protocol):
    def __call__(self, params : dict[ParameterKey, ParameterLike], data : Data, *loss_args) -> float:
        raise NotImplementedError

@runtime_checkable
class SpaceLike(Protocol):
    upper : ArrayLike
    lower : ArrayLike

@runtime_checkable
class BinnedModelLike(Protocol):
    def __call__(self, 
        params : dict[ParameterKey, ParameterLike]
    ) -> Data:
        raise NotImplementedError

@runtime_checkable
class UnbinnedModelLike(Protocol):
    def __call__(self, 
        x : ArrayLike, 
        params : dict[ParameterKey, ParameterLike]
    ) -> float:
        raise NotImplementedError


# class InternalParameterCollection(NamedTuple):
#     values : pt.PyTree[np.ndarray[float]]
#     uppers : pt.PyTree[np.ndarray[float]]
#     lowers : pt.PyTree[np.ndarray[float]]
#     floatings : pt.PyTree[bool]
        
@runtime_checkable
class MinimumLike(Protocol):
    valid : bool
    fmin : float
    params : dict[ParameterKey, float | ArrayLike]

@runtime_checkable
class MinimizerLike(Protocol):
    def minimize(self, loss : LossLike, params : dict[ParameterKey, ParameterLike], *loss_args, data : Data = None) -> MinimumLike:
        raise NotImplementedError

class InternalParameter:
    def __init__(self, external_param : ParameterLike, trusting : bool = False):
        if not trusting:
            if not isinstance(external_param, ParameterLike):
                raise ValueError("{} is not ParameterLike".format(external_param))
            try:
                np.broadcast_shapes(
                    np.shape(external_param.value), 
                    np.shape(external_param.upper), 
                    np.shape(external_param.lower), 
                    np.shape(external_param.floating)
                )
            except ValueError:
                raise ValueError("upper, lower, and floating must be broadcastable to value")
        
        self.value = external_param.value
        self.upper = external_param.upper
        self.lower = external_param.lower
        self.floating = external_param.floating

# def convert_params(params : pt.PyTree[ParameterLike]) -> InternalParameterCollection:
#     def format_values(p):
#         if not np.all(np.isfinite(p.value)):
#             raise ValueError("Invalid values in initial parameter values")
#         return np.array(p.value)

#     def format_lowers(p):
#         if np.shape(p.lower) != np.shape(p.upper):
#             raise ValueError("Lower and upper bounds on parameters have mismatching shape")
#         if hasattr(p.lower, "__len__"):
#             if np.shape(p.lower) != np.shape(p.value):
#                 raise ValueError("If bounds are provided array-like, it must be the same shape as values")
#             return np.array(p.lower)
#         return np.broadcast_to(p.lower, np.shape(p.value))

#     def format_floatings(p):
#         if not isinstance(p.floating, bool):
#             raise ValueError("floating must be bool")
#         return np.broadcast_to(p.floating, np.shape(p.value))
    
#     values = pt.map(format_values, params, is_leaf=lambda x: isinstance(x, ParameterLike))
#     lowers = pt.map(format_lowers, params, is_leaf=lambda x: isinstance(x, ParameterLike))
#     uppers = pt.map(
#         (lambda p: np.array(p.upper) if hasattr(p.upper, "__len__") else np.broadcast_to(p.upper, np.shape(p.value))),
#         params,
#         is_leaf=lambda x: isinstance(x, ParameterLike)
#     )
#     floatings = pt.map(format_floatings, params, is_leaf=lambda x: isinstance(x, ParameterLike))
#     return InternalParameterCollection(values, uppers, lowers, floatings)

# def convert_params(params : pt.PyTree[ParameterLike]) -> InternalParameterCollection:
#     good_bound = lambda x: x is not None and np.isfinite(x)
#     isarr = lambda x: hasattr(x, "__len__")

#     def set_default(value, lower, upper):
#         if value is None:
#             if good_bound(lower) and good_bound(upper):
#                 return np.mean((lower, upper))
#             elif good_bound(lower):
#                 return 0.0 if 0.0 > lower else lower + 1.0
#             elif good_bound(p.upper):
#                 return 0.0 if 0.0 < upper else upper - 1.0
#             else:
#                 return 0.0
#         elif not isinstance(value, float) or not np.isfinite(value):
#             raise ValueError("Parameter values not all valid finite floats")
#         else:
#             return value

#     def convert_value(p):
#         if not isinstance(p, ParameterLike):
#             raise ValueError("{} is not ParameterLike".format(p))
#         if not ((not isarr(p.value) and not isarr(p.lower) and not isarr(p.upper)) or (isarr(p.value) and isarr(p.lower) and isarr(p.upper))):
#             raise ValueError("value, upper, and lower must all be ArrayLike or all not ArrayLike")
            
#         if isarr(p.value):
#             if not (len(p.value) == len(p.upper) == len(p.lower)):
#                 raise ValueError("value, upper, and lower must have the same shape")
                
#             if hasattr(p.value, "__array__"):
#                 to_original = np.array
#             elif isinstance(p.value, list) or isinstance(p.value, tuple):
#                 to_original = type(p.value)
#             else:
#                 warnings.warn("Unrecognized type: {}. Defaulting to list.".format(type(p.value)))
#                 to_original = list
            
#             return pt.PVList((set_default(v, l, u) for v, l, u in zip(p.value, p.lower, p.upper)), to_original=to_original)
#         else:
#             return pt.PVList(set_default(p.value, p.lower, p.upper))

#     def convert_bounds(p):
#         if isarr(p.upper):
#             return pt.PVList((Bound(l, u) for l, u in zip(p.lower, p.upper)), to_original=list)
#         else:
#             return pt.PVList([Bound(p.lower, p.upper)], to_original=lambda x: x[0])
            

#     values = pt.map(convert_value, params, is_leaf = lambda x: isinstance(x, ParameterLike))
#     bounds = pt.map(convert_bounds, params, is_leaf = lambda x: isinstance(x, ParameterLike))
#     floatings = pt.map(
#         lambda p: pt.PVList([p.floating] * len(p.value)) if isarr(p.value) else pt.PVList(p.floating),
#         params,
#         is_leaf = lambda x: isinstance(x, ParameterLike)
#     )

#     return InternalParameterCollection(values=values, bounds=bounds, floatings=floatings)
    

# def convert_params(params : pt.PyTree[ParameterLike]) -> pt.PyTree[InternalParameter]:
#     """
#     Converts user parameters to InteralParameters.
#     Gives default values if not provided.
#     """
#     good_bound = lambda x: x is not None and np.isfinite(x)
#     isarr = lambda x: hasattr(x, "__len__")

#     def set_default(value, lower, upper):
#         if value is None:
#             if good_bound(lower) and good_bound(upper):
#                 return np.mean((lower, upper))
#             elif good_bound(lower):
#                 return 0.0 if 0.0 > lower else lower + 1.0
#             elif good_bound(p.upper):
#                 return 0.0 if 0.0 < upper else upper - 1.0
#             else:
#                 return 0.0
#         elif not isinstance(value, float) or not np.isfinite(value):
#             raise ValueError("Parameter values not all valid finite floats")
#         else:
#             return value
    
#     def convert(p):
#         if not isinstance(p, ParameterLike):
#             raise ValueError("{} is not ParameterLike".format(p))
#         if not ((not isarr(p.value) and not isarr(p.lower) and not isarr(p.upper)) or (isarr(p.value) and isarr(p.lower) and isarr(p.upper))):
#             raise ValueError("value, upper, and lower must all be ArrayLike or all not ArrayLike")
            
#         if isarr(p.value):
#             if not (len(p.value) == len(p.upper) == len(p.lower)):
#                 raise ValueError("value, upper, and lower must have the same shape")    
#             return InternalParameter(
#                 value = [set_default(v, l, u) for v, l, u in zip(p.value, p.lower, p.upper)],
#                 bounds = zip(p.lower, p.upper),
#                 floating = p.floating,
#                 to_original_type = np.array if hasattr(p.value, __array__) else list
#             )
#         else:
#             return InternalParameter(
#                 value = set_default(p.value, p.lower, p.upper), 
#                 bounds = (p.lower, p.upper), 
#                 floating = p.floating,
#                 to_original_type = float
#             )

#     return pt.map(convert, params, is_leaf = lambda x: isinstance(x, ParameterLike))
