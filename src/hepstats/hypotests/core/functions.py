from scipy.stats import norm

from ..calculators.basecalculator import BaseCalculator
from ...utils import api, POI

def _api_check(calculator, poinull, poialt = None):
    if not isinstance(calculator, BaseCalculator):
        raise ValueError("calculator must be Calculator")

    calculator.check_pois(poinull)
    if poialt:
        calculator.check_pois(poialt)

def discovery(calculator : BaseCalculator, poinull_path : api.ParameterPathLike, poinull_value : float = 0):
    poinull = POI(poinull_path, poinull_value)
    _api_check(calculator, poinull)
    if poinull.param_path(calculator.parameters.values).ndim > 0:
        raise ValueError("Discovery test only supported for scalar valued parameters")

    pnull, _ = calculator.pvalue(poinull, onesideddiscovery=True)
    pnull = pnull[0]

    significance = norm.ppf(1.0 - pnull)
    
    return pnull, significance