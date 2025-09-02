from scipy.stats import norm
from scipy import interpolate
import numpy as np
from numpy.typing import ArrayLike

from ..calculators.basecalculator import BaseCalculator
from ...utils import api, POI, POIarray

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
    calculator : BaseCalculator,
    poi_key : api.ParameterKey,
    poi_range : tuple = None,
    poi_altvalue : float = 0,
    alpha : float = 0.05,
    qtilde : bool = False,
    CLs : bool = True,
    ntests : int = 20
):
    if poi_range is not None and not len(poi_range) == 2:
        raise ValueError("POI range must be tuple of two values")

    reference_param = calculator.parameters[poi_key]
    if np.shape(reference_param.value) != tuple():
        raise ValueError("Upper limit test only supported for scalar valued parameters")
    if poi_range is None:
        poi_range = (reference_param.lower, reference_param.upper)

    poinull = POIarray(poi_key, np.linspace(*poi_range, ntests))
    poialt = POI(poi_key, poi_altvalue)
        
    _api_check(calculator, poinull, poialt)

    def pvalues(self, CLs: int = True) -> dict[str, np.ndarray]:
        """
        Returns p-values scanned for the values of the parameters of interest
        in the null hypothesis.

        Args:
            CLs: if `True` uses pvalues as :math:`p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}`
               else as :math:`p_{clsb} = p_{null}`.

        Returns:
            Dictionary of p-values for CLsb, CLs, expected (+/- sigma bands).
        """
        pnull, palt = calculator.pvalue(poinull=poinull, poialt=poialt, qtilde=qtilde, onesided=True)

        pvalues = {"clsb": pnull, "clb": palt}

        sigmas = [0.0, 1.0, 2.0, -1.0, -2.0]

        result = calculator.expected_pvalue(
            poinull=poinull,
            poialt=poialt,
            nsigma=sigmas,
            CLs=CLs,
            qtilde=qtilde,
            onesided=True,
        )

        pvalues["expected"] = result[0]
        pvalues["expected_p1"] = result[1]
        pvalues["expected_p2"] = result[2]
        pvalues["expected_m1"] = result[3]
        pvalues["expected_m2"] = result[4]

        pvalues["cls"] = pnull / palt

        return pvalues

    # create a filter for -1 and -2 sigma expected limits
    bestfit_poi_value = calculator.bestfit.params[poinull.param_key]
    under_fluc_filter = poinull.values >= bestfit_poi_value

    observed_key = "cls" if CLs else "clsb"

    to_interpolate = [observed_key] + [f"expected{i}" for i in ["", "_p1", "_m1", "_p2", "_m2"]]

    limits: dict = {}

    all_pvalues = pvalues(CLs)
    for k in to_interpolate:
        pvalues = all_pvalues[k]
        values = poinull.values

        if k == observed_key:
            k = "observed"
            pvalues = pvalues[under_fluc_filter]
            values = values[under_fluc_filter]

        if min(pvalues) > alpha:
            if k in ["expected", "observed"]:
                msg = f"The minimum of the scanned p-values is {min(pvalues)} which is larger than the"
                msg += f" confidence level alpha = {alpha}. Try to increase the maximum POI value."
                raise POIRangeError(msg)

            limits[k] = None
            continue

        tck = interpolate.splrep(values, pvalues - alpha, s=0)
        root = interpolate.sproot(tck)

        if len(root) > 1:
            root = root[0]

        try:
            limits[k] = float(root)
        except TypeError:
            limits[k] = None

    return limits, all_pvalues