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

from __future__ import annotations

import warnings

import uhi.typing.plottable

# def is_valid_parameter(object):
#     """
#     Checks if a parameter has the following attributes/methods:
#         * value
#         * set_value
#         * floating
#     """
#     has_value = hasattr(object, "value")
#     has_set_value = hasattr(object, "set_value")
#     has_floating = hasattr(object, "floating")

#     return has_value and has_set_value and has_floating


# def is_valid_data(object):
#     """
#     Checks if the data object has the following attributes/methods:
#         * nevents
#         * weights
#         * set_weights
#         * space
#     """
#     is_sampled_data = hasattr(object, "resample")

#     try:
#         has_nevents = hasattr(object, "nevents")
#     except RuntimeError:
#         if is_sampled_data:
#             object.resample()
#             has_nevents = hasattr(object, "nevents")
#         else:
#             has_nevents = False

#     has_weights = hasattr(object, "weights")
#     has_set_weights = hasattr(object, "set_weights")
#     has_space = hasattr(object, "space")
#     is_histlike = isinstance(object, uhi.typing.plottable.PlottableHistogram)
#     return (has_nevents and has_weights and has_set_weights and has_space) or is_histlike


# def is_valid_pdf(object):
#     """
#     Checks if the pdf object has the following attributes/methods:
#         * get_params
#         * pdf
#         * integrate
#         * sample
#         * get_yield

#     Also the function **is_valid_parameter** is called with each of the parameters returned by get_params
#     as argument.
#     """
#     has_get_params = hasattr(object, "get_params")
#     if not has_get_params:
#         return False
#     else:
#         params = object.get_params()

#     all_valid_params = all(is_valid_parameter(p) for p in params)
#     has_pdf = hasattr(object, "pdf")
#     has_integrate = hasattr(object, "integrate")
#     has_sample = hasattr(object, "sample")
#     has_space = hasattr(object, "space")
#     has_get_yield = hasattr(object, "get_yield")

#     return all_valid_params and has_pdf and has_integrate and has_sample and has_space and has_get_yield


# def is_valid_loss(object):
#     """
#     Checks if the loss object has the following attributes/methods:
#         * model
#         * data
#         * get_params
#         * constraints
#         * fit_range

#     Also the function **is_valid_pdf** is called with each of the models returned by model
#     as argument. Additionnally the function **is_valid_data** is called with each of the data objects
#     return by data as argument.
#     """
#     if not hasattr(object, "model"):
#         return False
#     else:
#         model = object.model

#     if not hasattr(object, "data"):
#         return False
#     else:
#         data = object.data

#     has_get_params = hasattr(object, "get_params")
#     has_constraints = hasattr(object, "constraints")
#     has_create_new = hasattr(object, "create_new")
#     if not has_create_new:
#         warnings.warn("Loss should have a `create_new` method.", FutureWarning, stacklevel=3)
#         has_create_new = True  # TODO: allowed now, will be dropped in the future
#     all_valid_pdfs = all(is_valid_pdf(m) for m in model)
#     all_valid_datasets = all(is_valid_data(d) for d in data)

#     return all_valid_pdfs and all_valid_datasets and has_constraints and has_create_new and has_get_params


# def is_valid_fitresult(object):
#     """
#     Checks if the fit result object has the following attributes/methods:
#         * loss
#         * params
#         * covariance

#     Also the function **is_valid_loss** is called with the loss as argument.
#     """
#     has_loss = hasattr(object, "loss")

#     if not has_loss:
#         return False
#     else:
#         loss = object.loss
#         has_params = hasattr(object, "params")
#         has_covariance = hasattr(object, "covariance")
#         return is_valid_loss(loss) and has_params and has_covariance


# def is_valid_minimizer(object):
#     """
#     Checks if the minimzer object has the following attributes/methods:
#         * minimize
#     """
#     return hasattr(object, "minimize")

from typing import runtime_checkable, Protocol, TypeVar, Callable, Union, NamedTuple
from numpy.typing import ArrayLike
import numpy as np
import warnings

from .pytree import pt

Data = TypeVar("Data", bound=pt.PyTree[ArrayLike])
T = TypeVar("T")

@runtime_checkable
class ParameterLike(Protocol):
    value : float | ArrayLike
    upper : float | ArrayLike
    lower : float | ArrayLike
    floating : bool

@runtime_checkable
class LossLike(Protocol):
    def __call__(self, params : pt.PyTree[float | ArrayLike], data : Data, *loss_args) -> float:
        raise NotImplementedError

@runtime_checkable
class ParameterPathLike(Protocol):
    def __call__(self, params : pt.PyTree[T]) -> T:
        raise NotImplementedError


@runtime_checkable
class SpaceLike(Protocol):
    upper : ArrayLike
    lower : ArrayLike

@runtime_checkable
class BinnedModelLike(Protocol):
    def __call__(self, 
        params : pt.PyTree[float | ArrayLike]
    ) -> Data:
        raise NotImplementedError

@runtime_checkable
class UnbinnedModelLike(Protocol):
    def __call__(self, 
        x : ArrayLike, 
        params : pt.PyTree[float | ArrayLike]
    ) -> float:
        raise NotImplementedError


class InternalParameterCollection(NamedTuple):
    values : pt.PyTree[np.ndarray[float]]
    uppers : pt.PyTree[np.ndarray[float]]
    lowers : pt.PyTree[np.ndarray[float]]
    floatings : pt.PyTree[bool]
        
@runtime_checkable
class MinimumLike(Protocol):
    valid : bool
    fmin : float
    params : pt.PyTree[Value]

@runtime_checkable
class MinimizerLike(Protocol):
    def minimize(self, loss : LossLike, params : InternalParameterCollection, *loss_args, data : Data = None) -> MinimumLike:
        raise NotImplementedError

# class ExpandableArray(NamedTuple):
#     value : float
#     shape : tuple[int]

#     def expand(self):
#         return np.broadcast_to(self.value, self.shape)

def convert_params(params : pt.PyTree[ParameterLike]) -> InternalParameterCollection:
    def format_values(p):
        if not np.all(np.isfinite(p.value)):
            raise ValueError("Invalid values in initial parameter values")
        return np.array(p.value)

    def format_lowers(p):
        if np.shape(p.lower) != np.shape(p.upper):
            raise ValueError("Lower and upper bounds on parameters have mismatching shape")
        if hasattr(p.lower, "__len__"):
            if np.shape(p.lower) != np.shape(p.value):
                raise ValueError("If bounds are provided array-like, it must be the same shape as values")
            return np.array(p.lower)
        return np.broadcast_to(p.lower, np.shape(p.value))

    def format_floatings(p):
        if not isinstance(p.floating, bool):
            raise ValueError("floating must be bool")
        return np.broadcast_to(p.floating, np.shape(p.value))
    
    values = pt.map(format_values, params, is_leaf=lambda x: isinstance(x, ParameterLike))
    lowers = pt.map(format_lowers, params, is_leaf=lambda x: isinstance(x, ParameterLike))
    uppers = pt.map(
        (lambda p: np.array(p.upper) if hasattr(p.upper, "__len__") else np.broadcast_to(p.upper, np.shape(p.value))),
        params,
        is_leaf=lambda x: isinstance(x, ParameterLike)
    )
    floatings = pt.map(format_floatings, params, is_leaf=lambda x: isinstance(x, ParameterLike))
    return InternalParameterCollection(values, uppers, lowers, floatings)

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
