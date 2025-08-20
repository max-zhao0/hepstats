from scipy.stats import norm
import numpy as np
from numpy.typing import ArrayLike

from ..calculators.basecalculator import BaseCalculator
from ...utils import api, POI

def _api_check(calculator, poinull, poialt = None):
    if not isinstance(calculator, BaseCalculator):
        raise ValueError("calculator must be Calculator")

    calculator.check_pois(poinull)
    if poialt:
        calculator.check_pois(poialt)

def discovery(
    calculator : BaseCalculator, 
    poinull_key : api.ParameterKey, 
    poinull_value : float = 0
):
    poinull = POI(poinull_key, poinull_value)
    _api_check(calculator, poinull)
    if np.shape(calculator.parameters[poinull.param_key].value) != tuple():
        raise ValueError("Discovery test only supported for scalar valued parameters")

    pnull, _ = calculator.pvalue(poinull, onesideddiscovery=True)
    pnull = pnull[0]

    significance = norm.ppf(1.0 - pnull)
    
    return pnull, significance

def upperlimit(
    calculator: BaseCalculator,
    poinull_key: api.ParameterKey,
    poinull_key: float | ArrayLike,
    poialt_key: api.ParameterKey,
    poialt_key: float,
    qtilde: bool = False
):
    
    _api_check(calculator, poinull, poialt)