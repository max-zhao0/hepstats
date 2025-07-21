from __future__ import annotations

from collections.abc import Mapping, Collection
# from contextlib import ExitStack, contextmanager, suppress

import numpy as np
from .. import api
from ..pytree import pt
from .poi import POI
from collections import OrderedDict

def get_ndims(dataset):
    """Return the number of dimensions in the dataset"""
    return len(dataset.obs)


def get_value(value):
    return np.array(value)


def set_values_once(params, values):
    with suppress(ImportError):
        import zfit

        return zfit.param.set_values(params, values)  # more efficient

    for p, v in zip(params, values):
        p.set_value(v)
    return None


def eval_pdf(model, x, params=None, allow_extended=False):
    """Compute pdf of model at a given point x and for given parameters values"""

    if params is None:
        params = {}

    def pdf(model, x):
        ret = model.ext_pdf(x) if model.is_extended and allow_extended else model.pdf(x)

        return get_value(ret)

    with ExitStack() as stack:
        for param in model.get_params():
            if param in params:
                value = params[param]["value"]
                stack.enter_context(param.set_value(value))
        return pdf(model, x)


def pll(pois : POI | Collection[POI],
    minimizer : api.MinimizerLike, 
    loss : api.LossLike, 
    params : api.InternalParameterCollection,
    *loss_args,
    data : api.Data = None,
    init=None
) -> float:
    """Compute minimum profile likelihood for fixed given parameters values."""
    del init  # unused currently
    if isinstance(pois, POI):
        pois = [pois]

    access_pois = lambda tree: tuple(p.param_path(tree) for p in pois)
    new_values = pt.at(access_pois, params.values, [p.value for p in pois])
    new_floatings = pt.at(access_pois, params.floatings, replace_fn=lambda f: np.broadcast_to(False, np.shape(f)))
    
    new_params = api.InternalParameterCollection(values=new_values, lowers=params.lowers, uppers=params.uppers, floatings=new_floatings)

    if pt.any(new_params.floatings):
        minimum = minimizer.minimize(loss, new_params, *loss_args, data=data)
        return minimum.fmin
    else:
        return loss(new_params.values, data, *loss_args) if data is not None else loss(new_params.values, *loss_args)

    # with ExitStack() as stack:
    #     for p in pois:
    #         param = p.parameter
    #         stack.enter_context(param.set_value(p.value))
    #         param.floating = False

    #     if any(param_loss.floating for param_loss in loss.get_params()):
    #         minimum = minimizer.minimize(loss=loss)  # TODO: add init?
    #         value = minimum.fmin
    #     else:
    #         value = get_value(loss.value())

    #     for p in pois:
    #         p.parameter.floating = True

    # pspecs, spec_treedef = pt.tree_flatten(loss.get_param_specs(), is_leaf=lambda x: isinstance(x, ParameterSpecLike))
    # spec_value_map = OrderedDict()
    # for spec in pspecs:
    #     spec_value_map[spec] = ParameterValue()
    
    # assert len(spec_value_map) == len(pspecs), "Ensure that __hash__ and __eq__ for parameter specs do not collide in provided specs"

    # for p in pois:
    #     spec_value_map[p.param_spec] = ParameterValue(value=p.value, floating=False)

    # pvalues = list(spec_value_map.values())

    # minimum = minimizer.minimize(loss, pt.tree_unflatten(spec_treedef, pvalues))
    # return minimum.fmin

# @contextmanager
def set_values(params, values=None):
    pass
#     if values is None:
#         if isinstance(params, Mapping):
#             values = tuple(params.values())
#             params = tuple(params.keys())
#         else:
#             msg = "values must be provided if params is not a Mapping (dict-like)"
#             raise ValueError(msg)
#     old_values = [p.value() for p in params]
#     for p, v in zip(params, values):
#         p.set_value(v)
#     yield
#     for p, v in zip(params, old_values):
#         p.set_value(v)


def array2dataset(dataset_cls, obs, array, weights=None):
    """
    dataset_cls: only used to get the class in which array/weights will be
    converted.
    """

    if hasattr(dataset_cls, "from_numpy"):
        return dataset_cls.from_numpy(obs, array=array, weights=weights)
    else:
        return dataset_cls(obs, array=array, weights=weights)


def get_nevents(dataset):
    """Returns the number of events in the dataset"""

    return get_value(dataset.nevents)
