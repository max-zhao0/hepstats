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

@runtime_checkable
class ParameterLike(Protocol):
    value : float | ArrayLike
    upper : float | ArrayLike
    lower : float | ArrayLike
    floating : bool

@runtime_checkable
class LossLike(Protocol):
    def __call__(self, params : dict[ParameterKey, float | ArrayLike], *loss_args) -> float:
        raise NotImplementedError

@runtime_checkable
class NegativeLogLikelihoodLike(Protocol):
    def __call__(self, params : dict[ParameterKey, float | ArrayLike], *nll_args, data : dict[DataKey, ArrayLike]) -> float:
        raise NotImplementedError

@runtime_checkable
class BinnedModelLike(Protocol):
    def expected_histogram(self, 
        params : dict[ParameterKey, float | ArrayLike]
    ) -> ArrayLike:
        raise NotImplementedError

@runtime_checkable
class UnbinnedModelLike(Protocol):
    lower : float | ArrayLike
    upper : float | ArrayLike
    
    def pdf(self, 
        x : float | ArrayLike, 
        params : dict[ParameterKey, float | ArrayLike]
    ) -> float:
        raise NotImplementedError

@runtime_checkable
class ExtendedUnbinnedModelLike(UnbinnedModelLike, Protocol):
    def get_yield(self, params : dict[ParameterKey, float | ArrayLike]) -> float:
        raise NotImplementedError

@runtime_checkable
class UnextendedUnbinnedModelLike(UnbinnedModelLike, Protocol):
    N : int

ModelLike = TypeVar("ModelLike", bound=BinnedModelLike | UnbinnedModelLike)
        
@runtime_checkable
class MinimumLike(Protocol):
    valid : bool
    fmin : float
    params : dict[ParameterKey, float | ArrayLike]

@runtime_checkable
class MinimizerLike(Protocol):
    def minimize(self, loss : LossLike, params : dict[ParameterKey, ParameterLike], *loss_args) -> MinimumLike:
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
