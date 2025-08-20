from __future__ import annotations

# from collections.abc import Callable
from typing import Any, Callable, Iterable
import numpy as np

from ...utils import base_sample, base_sampler, pll, api, POIarray, POI
from ..hypotests_object import HypotestsObject
# from ..parameters import POI, POIarray, asarray
from ..toyutils import ToyResult, ToysManager


class BaseCalculator(HypotestsObject):
    """Base class for calculator."""

    def __init__(self, 
        loss : api.LossLike, 
        params : dict[api.ParameterKey, api.ParameterLike], 
        *loss_args,
        data : api.Data = None,
        minimizer : api.MinimizerLike = None,
        blind : bool = True,
        **kwargs
    ):
        """
        Args:
            input: loss or fit result
            minimizer: minimizer to use to find the minimum of the loss function

        Example with **zfit**:
            >>> import zfit
            >>> from zfit.core.loss import UnbinnedNLL
            >>> from zfit.minimize import Minuit
            >>>
            >>> obs = zfit.Space('x', limits=(0.1, 2.0))
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> sigma = zfit.Parameter("sigma", 0.1)
            >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> loss = UnbinnedNLL(model=model, data=data)
            >>>
            >>> calc = BaseCalculator(input=loss, minimizer=Minuit())
        """
        super().__init__(loss, params, *loss_args, data=None, minimizer=minimizer, **kwargs)
        
        self._blind = blind
        self._obs_nll = {}

        # self._parameters = {}
        # for m in self.model:
        #     for d in m.get_params():
        #         self._parameters[d.name] = d

    @property
    def blind(self):
        return self._blind

    @blind.setter
    def blind(self, blind : bool):
        self._blind = blind

    def obs_nll(self, pois: POIarray) -> np.ndarray:
        """Compute observed negative log-likelihood values for given parameters of interest.

        Args:
            pois: parameters of interest.

        Returns:
            Observed nll values.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poi = POI(mean, [1.1, 1.2, 1.0])
            >>> nll = calc.obs_nll(poi)
        """
    
        # ret = np.empty(pois.shape)
        # for i, p in enumerate(pois):
        #     if p not in self._obs_nll:
        #         nll = pll(minimizer=self.minimizer, loss=self.loss, pois=p)
        #         self._obs_nll[p] = nll
        #     ret[i] = self._obs_nll[p]
        # return ret

        ret = np.empty(pois.shape)
        for i, p in enumerate(pois):
            ret[i] = pll(p, self.minimizer, self.loss, self.parameters, *self.loss_args, data=self.data)
        return ret

    def qobs(
        self,
        poinull: POI,
        onesided: bool = True,
        onesideddiscovery: bool = True,
        qtilde: bool = False,
    ):
        """Computes observed values of the :math:`\\Delta` log-likelihood test statistic.

        Args:
            poinull: parameters of interest for the null hypothesis.
            qtilde: if `True` use the :math:`\\tilde{q}` test statistics else (default)
              use the :math:`q` test statistic.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a
              discovery test.

        Returns:
            Observed values of q.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poi = POI(mean, [1.1, 1.2, 1.0])
            >>> q = calc.qobs(poi)
        """

        self.check_pois(poinull)
        assert isinstance(poinull, POI)

        # if poinull.ndim == 1:
        #     poi_spec = poinull.param_spec
        #     bestfitpoi_val = self.bestfit.spec_value_map[poi_spec]

        #     if qtilde and bestfit < 0:
        #         bestfitpoi = POI(poi_spec, 0)
        #     else:
        #         bestfitpoi = POI(poi_spec, bestfitpoi_val)
        #         self._obs_nll[bestfitpoi] = self.bestfit.fmin

        bestfitpoi_val = self.bestfit.params[poinull.param_key]
        assert np.shape(bestfitpoi_val) == tuple(), "qobs cannot be calculated with array valued POI"
        
        if qtilde:
            bestfitpoi_val = max(bestfitpoi_val, 0)
        bestfitpoi = POI(poinull.param_key, bestfitpoi_val)
        
        nll_bestfitpoi_obs = self.obs_nll(bestfitpoi)
        nll_poinull_obs = self.obs_nll(poinull)

        print("nll_poinull_obs={}, nll_bestfitpoi_obs={}".format(nll_poinull_obs, nll_bestfitpoi_obs))
        print("poinull={}, bestfitpoi={}".format(poinull, bestfitpoi))
        
        return self.q(
            nll1=nll_poinull_obs,
            nll2=nll_bestfitpoi_obs,
            poi1=poinull,
            poi2=bestfitpoi,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def pvalue(
        self,
        poinull: POI | POIarray,
        poialt: POI | None = None,
        qtilde: bool = False,
        onesided: bool = True,
        onesideddiscovery: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes pvalues for the null and alternative hypothesis.

        Args:
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery test.

        Returns:
            Tuple of arrays for pnull, palt

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> pvalues = calc.pavalue(poinull, poialt)
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._pvalue_(
            poinull=poinull,
            poialt=poialt,
            qtilde=qtilde,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    def expected_pvalue(
        self,
        poinull: POI | POIarray,
        poialt: POI | POIarray,
        nsigma: list[int],
        CLs: bool = False,
        qtilde: bool = False,
        onesided: bool = True,
        onesideddiscovery: bool = False,
    ) -> list[np.array]:
        """Computes the expected pvalues and error bands for different values of :math:`\\sigma` (0=expected/median)

        Args:
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            nsigma: list of values of :math:`\\sigma` to compute the expected pvalue.
            CLs: if `True` computes pvalues as :math:`p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}`
              else as :math:`p_{clsb} = p_{null}`.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery.

        Returns:
            Array of expected pvalues for each :math:`\\sigma` value

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.expected_pvalue(poinull, poialt)
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        if qtilde and (poialt.values < 0).any():
            poialt = POIarray(
                parameter=poialt.parameter,
                values=np.where(poialt.values < 0, 0, poialt.values),
            )

        return self._expected_pvalue_(
            poinull=poinull,
            poialt=poialt,
            nsigma=nsigma,
            CLs=CLs,
            qtilde=qtilde,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, qtilde, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError
    
    def check_pois(self, pois: POI):
        """
        Checks if the parameter of interest is a :class:`hepstats.parameters.POIarray` instance.

        Args:
            pois: the parameter of interest to check.

        Raises:
            TypeError: if pois is not an instance of :class:`hepstats.parameters.POIarray`.
        """

        # msg = "POI/POIarray is required."
        # if not isinstance(pois, POIarray):
        #     raise TypeError(msg)
        # if pois.ndim > 1:
        #     msg = "Tests with more that one parameter of interest are not yet implemented."
        #     raise NotImplementedError(msg)

        if not isinstance(pois, POIarray):
            raise ValueError("Not POI or POIarray")

        if pois.param_key not in self.parameters:
            raise ValueError("{} is not in calculator's parameters".format(pois.param_key))
        reference_param = self.parameters[pois.param_key]

        # if not isinstance(reference_param, api.ParameterLike):
        #     raise ValueError("{} needs to point to a Parameter".format(param_key))
        if not reference_param.floating:
            raise ValueError("POI must point to a floating parameter")
        if np.shape(reference_param.value) != np.shape(pois.values)[1:]:
            raise ValueError("POI values provided do not match the shape of the parameter values it points to")

    @staticmethod
    def check_pois_compatibility(poi1: POI | POIarray, poi2: POI | POIarray):
        """
        Checks compatibility between two lists of :func:`hepstats.parameters.POIarray` instances.

        Args:
            poi1: the first parameter of interest.
            poi2: the second parameter of interest.

        Raises:
            ValueError: if the two parameters of interests don't have the same parameters, check by their
               names.
        """

        # if poi1.ndim != poi2.ndim:
        #     msg = f"POIs should have the same dimensions, poi1={poi1.ndim}, poi2={poi2.ndim}"
        #     raise ValueError(msg)

        # if poi1.ndim == 1 and poi1.name != poi2.name:
        #     msg = "The variables used in the parameters of interest should have the same names,"
        #     msg += f" poi1={poi1.name}, poi2={poi2.name}"
        #     raise ValueError(msg)

    def q(
        self,
        nll1: np.array,
        nll2: np.array,
        poi1: POIarray,
        poi2: POIarray,
        onesided: bool = True,
        onesideddiscovery: bool = False,
    ) -> np.ndarray:
        """Compute values of the test statistic q defined as the difference between negative log-likelihood
        values :math:`q = nll1 - nll2`.

        Args:
            nll1: array of nll values #1, evaluated with poi1.
            nll2: array of nll values #2, evaluated with poi2.
            poi1: POI's #1.
            poi2: POI's #2.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery.

        Returns:
            Array of :math:`q` values.
        """

        self.check_pois(poi1)
        self.check_pois(poi2)
        self.check_pois_compatibility(poi1, poi2)

        assert len(nll1) == len(poi1)
        assert len(nll2) == len(poi2)

        poi1_values = poi1.values
        poi2_values = poi2.values

        q = 2 * (nll1 - nll2)
        zeros = np.zeros(q.shape)

        if onesideddiscovery:
            condition = (poi2_values < poi1_values) | (q < 0)
        elif onesided:
            condition = (poi2_values > poi1_values) | (q < 0)
        else:
            condition = q < 0

        return np.where(condition, zeros, q)


class BaseToysCalculator(BaseCalculator):
    def __init__(self, input, minimizer, **kwargs):
        """Basis for toys calculator class.

        Args:
            input: loss or fit result.
            minimizer: minimizer to use to find the minimum of the loss function.
            sampler: function used to create sampler with models, number of events and floating
               parameters in the sample.
            sample: function used to get samples from the sampler.
        """
        super().__init__(input, minimizer, **kwargs)


class ToysCalculator(BaseToysCalculator, ToysManager):
    """
    Class for calculators using toys.
    """

    def __init__(
        self,
        input,
        minimizer,
        ntoysnull: int = 100,
        ntoysalt: int = 100,
        sampler: Callable = base_sampler,
        sample: Callable = base_sample,
    ):
        """Toys calculator class.

        Args:
            input: loss or fit result.
            minimizer: minimizer to use to find the minimum of the loss function.
            ntoysnull: minimum number of toys to generate for the null hypothesis.
            ntoysalt: minimum number of toys to generate for the alternative hypothesis.
            sampler: function used to create sampler with models, number of events and floating** parameters.
                in the sample Default is :func:`hepstats.utils.fit.sampling.base_sampler`.
            sample: function used to get samples from the sampler. Default is
                :func:`hepstats.utils.fit..sampling.base_sample`.
        """
        super().__init__(input, minimizer, sampler=sampler, sample=sample)

        self._ntoysnull = ntoysnull
        self._ntoysalt = ntoysalt

    @classmethod
    def from_yaml(
        cls,
        filename: str,
        input,
        minimizer,
        sampler: Callable = base_sampler,
        sample: Callable = base_sample,
        **kwargs: Any,
    ):
        """
        ToysCalculator constructor with the toys loaded from a yaml file.

        Args:
            filename: the yaml file name.
            input: loss or fit result.
            minimizer: minimizer to use to find the minimum of the loss function.
            sampler: function used to create sampler with models, number of events and floating
               parameters in the sample Default is :func:`hepstats.utils.fit.sampling.base_sampler`.
            sample: function used to get samples from the sampler. Default is
               :func:`hepstats.fitutils.sampling.base_sample`.
        """

        ntoysnull = kwargs.get("ntoysnull", 100)
        ntoysalt = kwargs.get("ntoysall", 100)

        calculator = cls(
            input=input,
            minimizer=minimizer,
            ntoysnull=ntoysnull,
            ntoysalt=ntoysalt,
            sampler=sampler,
            sample=sample,
        )
        toysresults = calculator.toysresults_from_yaml(filename)

        for t in toysresults:
            calculator.add_toyresult(t)

        return calculator

    @property
    def ntoysnull(self) -> int:
        """
        Returns the number of toys generated for the null hypothesis.
        """
        return self._ntoysnull

    @ntoysnull.setter
    def ntoysnull(self, n: int):
        """
        Set the number of toys generated for the null hypothesis.

        Args:
            n: number of toys
        """
        self._ntoysnull = n

    @property
    def ntoysalt(self) -> int:
        """
        Returns the number of toys generated for the alternative hypothesis.
        """
        return self._ntoysalt

    @ntoysalt.setter
    def ntoysalt(self, n: int):
        """
        Set the number of toys generated for the alternative hypothesis.

        Args:
            n: number of toys
        """
        self._ntoysalt = n

    def _get_toys(
        self,
        poigen: POI | POIarray,
        poieval: POI | POIarray | None = None,
        qtilde: bool = False,
        hypothesis: str = "null",
    ) -> dict[POI, ToyResult]:
        """
        Return the generated toys for a given POI.

        Args:
            poigen: POI used to generate the toys.
            poieval: POI values to evaluate the loss function.
            qtilde: if `True` use the :math:`\tilde{q}` test statistics else (default) use the :math:`q` test statistic.
            hypothesis: `null` or `alternative`.
        """

        if hypothesis not in {"null", "alternative"}:
            msg = "hypothesis must be 'null' or 'alternative'."
            raise ValueError(msg)

        ntoys = self.ntoysnull if hypothesis == "null" else self.ntoysalt

        ret = {}

        for p in poigen:
            if poieval is None:
                poieval_p = asarray(p)
            else:
                poieval_p = poieval
                if p not in poieval_p:
                    poieval_p = poieval_p.append(p.value)

            if qtilde and 0.0 not in poieval_p.values:
                poieval_p = poieval_p.append(0.0)

            ngenerated = self.ntoys(p, poieval_p)
            ntogen = ntoys - ngenerated if ngenerated < ntoys else 0

            if ntogen > 0:
                self.generate_and_fit_toys(ntoys=ntogen, poigen=p, poieval=poieval_p)

            ret[p] = self.get_toyresult(p, poieval_p)

        return ret

    def get_toys_null(
        self,
        poigen: POI | POIarray,
        poieval: POI | POIarray | None = None,
        qtilde: bool = False,
    ) -> dict[POI, ToyResult]:
        """
        Return the generated toys for the null hypothesis.

        Args:
            poigen: POI used to generate the toys.
            poieval: POI values to evaluate the loss function.
            qtilde: if `True` use the :math:`\tilde{q}` test statistics else (default) use the :math:`q` test statistic.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> for p in poinull:
            ...     calc.get_toys_alt(p, poieval=poialt)
        """
        return self._get_toys(poigen=poigen, poieval=poieval, qtilde=qtilde, hypothesis="null")

    def get_toys_alt(
        self,
        poigen: POI | POIarray,
        poieval: POI | POIarray | None = None,
        qtilde: bool = False,
    ) -> dict[POI, ToyResult]:
        """
        Return the generated toys for the alternative hypothesis.

        Args:
            poigen: POI used to generate the toys.
            poieval: POI values to evaluate the loss function.
            qtilde: if `True` use the :math:`\tilde{q}` test statistics else (default) use the :math:`q` test statistic.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> calc.get_toys_alt(poialt, poieval=poinull)
        """
        return self._get_toys(poigen=poigen, poieval=poieval, qtilde=qtilde, hypothesis="alternative")
