from __future__ import annotations

import math
import typing
import warnings
from typing import Any

from numpy.typing import ArrayLike
import numpy as np
from scipy.stats import norm

from ...utils import pll, api, sample
# from ...utils.fit.api_check import LossLike, MinimumLike, MinimizerLike # is_valid_fitresult, is_valid_loss
from ...utils.fit.diverse import get_ndims, sample
# from ..parameters import POI, POIarray
from .basecalculator import BaseCalculator


def generate_asimov_hist(
    model, params: dict[Any, dict[str, Any]], nbins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the Asimov histogram using a model and dictionary of parameters.

    Args:
        model: model used to generate the dataset.
        params: values of the parameters of the models.
        nbins: number of bins.

    Returns:
        Tuple of hist and bin_edges.

    Example with **zfit**:
        >>> obs = zfit.Space('x', limits=(0.1, 2.0))
        >>> mean = zfit.Parameter("mu", 1.2)
        >>> sigma = zfit.Parameter("sigma", 0.1)
        >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
        >>> hist, bin_edges = generate_asimov_hist(model, {"mean": 1.2, "sigma": 0.1})
    """
    if nbins is None:
        nbins = 100
    space = model.space
    bounds = space.limit1d
    bin_edges = np.linspace(*bounds, nbins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    hist = eval_pdf(model, bin_centers, params, allow_extended=True)
    hist *= space.area() / nbins

    return hist, bin_edges


def generate_asimov_dataset(data, model, is_binned, nbins, values):
    """Generate the Asimov dataset using a model and dictionary of parameters.

    Args:
        data: Data, the same class should be used for the generated dataset.
        model: Model to use for the generation. Can be binned or unbinned.
        is_binned: If the model is binned.
        nbins: Number of bins for the asimov dataset.
        values: Dictionary of parameters values.

    Returns:
        Dataset with the asimov dataset.
    """
    nsample = None
    if not model.is_extended:
        nsample = get_value(data.n_events)
    if is_binned:
        with set_values(list(values), [v["value"] for v in values.values()]):
            dataset = model.to_binneddata()
            if nsample is not None:
                dataset = type(dataset).from_hist(dataset.to_hist() * nsample)
    else:
        if len(nbins) > 1:  # meaning we have multiple dimensions
            msg = (
                "Currently, only one dimension is supported for models that do not follow"
                " the new binned loss convention. New losses can be registered with the"
                " asymtpotic calculator."
            )
            raise ValueError(msg)
        weights, bin_edges = generate_asimov_hist(model, values, nbins[0])
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        if nsample is not None:  # It's not extended
            weights *= nsample

        dataset = array2dataset(type(data), data.space, bin_centers, weights)
    return dataset


class AsymptoticCalculator(BaseCalculator):
    """
    Class for asymptotic calculators, using asymptotic formulae of the likelihood ratio described in
    :cite:`Cowan:2010js`. Can be used only with one parameter of interest.
    """

    UNBINNED_TO_BINNED_LOSS: typing.ClassVar = {}
    try:
        from zfit.loss import (
            BinnedNLL,
            ExtendedBinnedNLL,
            ExtendedUnbinnedNLL,
            UnbinnedNLL,
        )
    except ImportError:
        pass
    else:
        UNBINNED_TO_BINNED_LOSS[UnbinnedNLL] = BinnedNLL
        UNBINNED_TO_BINNED_LOSS[ExtendedUnbinnedNLL] = ExtendedBinnedNLL

    def __init__(self,
        nll : api.LossLike | api.NegativeLogLikelihoodlike, 
        params : dict[api.ParameterKey, api.ParameterLike], 
        *loss_args,
        data : dict[api.DataKey, ArrayLike] = None, # CAN BE NONE
        models : dict[api.DataKey, api.ModelLike] = None, # CAN BE NONE
        minimizer : api.MinimizerLike = None, 
        extended_unbinned_yield_correction : bool | dict[api.DataKey, bool] = True, # CANNOT BE NONE
        beta : int | dict[api.DataKey, int] = 100, # CANNOT BE NONE
        blind : bool = True,
        **kwargs
    ):
        """Asymptotic calculator class using Wilk's and Wald's asymptotic formulae.

        The asympotic formula is significantly faster than the Frequentist calculator, as it does not
        require the calculation of the frequentist p-value, which involves the calculation of toys (sample-and-fit).


        Args:
            input: loss or fit result.
            minimizer: minimizer to use to find the minimum of the loss function.
            asimov_bins: number of bins of the Asimov dataset.

        Example with **zfit**:
            >>> import zfit
            >>> from zfit.loss import UnbinnedNLL
            >>> from zfit.minimize import Minuit
            >>>
            >>> obs = zfit.Space('x', limits=(0.1, 2.0))
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> sigma = zfit.Parameter("sigma", 0.1)
            >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> loss = UnbinnedNLL(model=model, data=data)
            >>>
            >>> calc = AsymptoticCalculator(input=loss, minimizer=Minuit(), asimov_bins=100)
        """

        super().__init__(nll, params, *loss_args, data=data, models=models, minimizer=minimizer, blind=blind, **kwargs)

        self._unbinned_corr = self.format_datalike_dict(extended_unbinned_yield_correction, bool)
        self._beta = self.format_datalike_dict(beta, int)

        self._asimov_dataset: dict = {}
        self._asimov_norms : dict = {}
        self._asimov_loss: dict = {}
        self._binned_loss = None
        # cache of nll values computed with the asimov dataset
        self._asimov_nll: dict[POI, np.ndarray] = {}

    @property
    def extended_unbinned_yield_correction(self):
        return self._unbinned_corr

    @extended_unbinned_yield_correction.setter
    def extended_unbinned_yield_correction(self, value : bool | dict[api.DataKey, bool]):
        self._asimov_dataset: dict = {}
        self._asimov_norms : dict = {}
        self._asimov_loss: dict = {}
        self._unbinned_corr = self.format_datalike_dict(value, bool)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value : int | dict[api.DataKey, int]):
        self._asimov_dataset: dict = {}
        self._asimov_norms : dict = {}
        self._asimov_loss: dict = {}
        self._beta = self.format_datalike_dict(value, int)

    def asimov_dataset(self, poi: POI, ntrials_fit: int = 5):
        if poi not in self._asimov_dataset:
            assert poi not in self._asimov_norms
            
            if self.blind:
                # Use nominal values
                asimov_param_values = {k : self.parameters[k].value for k in self.parameters}
            else:
                # Fit over nuisances
                asimov_param_values = pll(poi, self.minimizer, self.loss, self.parameters, ntrials_fit=ntrials_fit).params
                
            asimov_dataset = {}
            asimov_norm = {}
            for data_key in self.data:
                model = self.models[data_key]
                beta = self.beta[data_key]

                if isinstance(model, api.BinnedModelLike):
                    raise NotImplementedError
                elif isinstance(model, api.UnbinnedModelLike):
                    norm = model.get_yield(asimov_param_values) if isinstance(model, api.ExtendedUnbinnedModelLike) else model.N

                    if not self.blind:
                        nsamples = int(beta * max(norm, len(self.data[data_key])))
                    else:
                        nsamples = int(beta * norm)

                    # Remember the normalization under Asimov parameters, to be used to correct the Asimov loss if necessary
                    asimov_norm[data_key] = norm
                    asimov_dataset[data_key] = sample(model, asimov_param_values, nsamples=nsamples, minimizer=self.minimizer)
                else:
                    raise NotImplementedError

            self._asimov_norms[poi] = asimov_norm
            self._asimov_dataset[poi] = asimov_dataset

        assert poi in self._asimov_norms
        return self._asimov_dataset[poi]

    def asimov_diagnostics(self, poi : POI):
        if self.blind:
            asimov_param_values = {k : self.parameters[k].value for k in self.parameters}
        else:
            asimov_param_values = pll(poi, self.minimizer, self.loss, self.parameters).params

        asimov_fitted_param_values = pll([], self.minimizer, self.asimov_loss(poi), self.parameters).params
        print("asimov_param_values = {}".format(asimov_param_values))
        print("asimov_fitted_param_values = {}".format(asimov_fitted_param_values))

    def asimov_norm(self, poi: POI):
        if poi not in self._asimov_norms:
            self.asimov_dataset(poi)
        return self._asimov_norms[poi]

    def asimov_loss(self, poi: POI):
        """Constructs a loss function using the Asimov dataset for a given alternative hypothesis.

        Args:
            poi: parameter of interest of the alternative hypothesis.

        Returns:
             Loss function.

        Example with **zfit**:
            >>> poialt = POI(mean, 1.2)
            >>> loss = calc.asimov_loss(poialt)
        """
        if self.models is None:
            raise ValueError("Performing a statistical procedure that requires models, but none provided to calculator")
            
        if poi not in self._asimov_loss:
            asimov_dataset = self.asimov_dataset(poi)
            asimov_norm = self.asimov_norm(poi)

            unbinned_corrections = []
            for data_key in asimov_dataset:
                model = self.models[data_key]
                
                if isinstance(model, api.UnbinnedModelLike):
                    # Downweight oversampling
                    asimov_size = len(asimov_dataset[data_key])
                    corr = lambda params: (1 - asimov_norm[data_key] / asimov_size) * np.sum(np.log(model.pdf(asimov_dataset[data_key], params)))
                    unbinned_corrections.append(corr)
                
                    if isinstance(model, api.ExtendedUnbinnedModelLike) and self._unbinned_corr[data_key]:
                        # Correct yield factor in unbinned asimov
                        unbinned_corrections.append(lambda params: (asimov_size - asimov_norm[data_key]) * np.log(model.get_yield(params)))
            
            loss = self.lossbuilder(self.nll, asimov_dataset, unbinned_corrections)
            self._asimov_loss[poi] = loss

        return self._asimov_loss[poi]

    def asimov_nll(self, pois: POIarray, poialt: POI) -> np.ndarray:
        """Computes negative log-likelihood values for given parameters of interest using the Asimov dataset
        generated with a given alternative hypothesis.

        Args:
            pois: parameters of interest.
            poialt: parameter of interest of the alternative hypothesis.

        Returns:
            Array of nll values for the alternative hypothesis.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.asimov_nll(poinull, poialt)

        """
        self.check_pois(pois)
        self.check_pois(poialt)

        ret = np.empty(pois.shape)
        for i, p in enumerate(pois):
            if p not in self._asimov_nll:
                nll = pll(p, self.minimizer, self.asimov_loss(poialt), self.parameters, *self.loss_args).fmin
                self._asimov_nll[p] = nll
            ret[i] = self._asimov_nll[p]
        return ret

    def pnull(
        self,
        qobs: np.ndarray,
        qalt: np.ndarray | None = None,
        onesided: bool = True,
        onesideddiscovery: bool = False,
        qtilde: bool = False,
        nsigma: int = 0,
    ) -> np.ndarray:
        """Computes the pvalue for the null hypothesis.

        Args:
            qobs: observed values of the test-statistic q.
            qalt: alternative values of the test-statistic q using the asimov dataset.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.
            nsigma: significance shift.

        Returns:
             Array of the pvalues for the null hypothesis.
        """
        sqrtqobs = np.sqrt(qobs)

        # 1 - norm.cdf(x) == norm.cdf(-x)
        if onesided or onesideddiscovery:
            pnull = 1.0 - norm.cdf(sqrtqobs - nsigma)
        else:
            pnull = (1.0 - norm.cdf(sqrtqobs - nsigma)) * 2.0

        if qalt is not None and qtilde:
            cond = (qobs > qalt) & (qalt > 0)
            sqrtqalt = np.sqrt(qalt)
            pnull_2 = 1.0 - norm.cdf((qobs + qalt) / (2.0 * sqrtqalt) - nsigma)

            if not (onesided or onesideddiscovery):
                pnull_2 += 1.0 - norm.cdf(sqrtqobs - nsigma)

            pnull = np.where(cond, pnull_2, pnull)

        # print("pnull = {}, qobs = {}".format(pnull, qobs))

        return pnull

    def qalt(
        self,
        poinull: POIarray,
        poialt: POI,
        onesided: bool,
        onesideddiscovery: bool,
        qtilde: bool = False,
    ) -> np.ndarray:
        """Computes alternative hypothesis values of the :math:`\\Delta` log-likelihood test statistic using the asimov
        dataset.

        Args:
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a
              discovery test.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.

        Returns:
            Q values for the alternative hypothesis.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, [1.2])
            >>> q = calc.qalt(poinull, poialt)
        """
        poialt_bf = POI(poialt.param_key, 0) if qtilde and poialt.value < 0 else poialt

        # print("poialt_bf = {}, poialt = {}, poinull = {}".format(poialt_bf, poialt, poinull))

        nll_poialt_asy = self.asimov_nll(poialt_bf, poialt)
        nll_poinull_asy = self.asimov_nll(poinull, poialt)

        # print("nll_poialt_asy = {}, nll_poinull_asy = {}".format(nll_poialt_asy, nll_poinull_asy))

        return self.q(
            nll1=nll_poinull_asy,
            nll2=nll_poialt_asy,
            poi1=poinull,
            poi2=poialt,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def palt(
        self,
        qobs: np.ndarray,
        qalt: np.ndarray,
        onesided: int = True,
        onesideddiscovery: int = False,
        qtilde: int = False,
    ) -> np.ndarray:
        """Computes the pvalue for the alternative hypothesis.

        Args:
            qobs: observed values of the test-statistic q.
            qalt: alternative values of the test-statistic q using the Asimov dataset.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.

        Returns:
             Array of the pvalues for the alternative hypothesis.
        """
        sqrtqobs = np.sqrt(qobs)
        sqrtqalt = np.sqrt(qalt)

        # 1 - norm.cdf(x) == norm.cdf(-x)
        if onesided or onesideddiscovery:
            palt = 1.0 - norm.cdf(sqrtqobs - sqrtqalt)
        else:
            palt = 1.0 - norm.cdf(sqrtqobs + sqrtqalt)
            palt += 1.0 - norm.cdf(sqrtqobs - sqrtqalt)

        if qtilde:
            cond = qobs > qalt
            palt_2 = 1.0 - norm.cdf((qobs - qalt) / (2.0 * sqrtqalt))

            if not (onesided or onesideddiscovery):
                palt_2 += 1.0 - norm.cdf(sqrtqobs + sqrtqalt)

            palt = np.where(cond, palt_2, palt)

        return palt

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):
        qobs = self.qobs(
            poinull,
            onesided=onesided,
            qtilde=qtilde,
            onesideddiscovery=onesideddiscovery,
        )

        needpalt = poialt is not None

        if needpalt:
            qalt = self.qalt(
                poinull=poinull,
                poialt=poialt,
                onesided=onesided,
                onesideddiscovery=onesideddiscovery,
                qtilde=qtilde,
            )
            palt = self.palt(
                qobs=qobs,
                qalt=qalt,
                onesided=onesided,
                qtilde=qtilde,
                onesideddiscovery=onesideddiscovery,
            )
        else:
            qalt = None
            palt = None

        # print("qalt = {}".format(qalt))

        pnull = self.pnull(
            qobs=qobs,
            qalt=qalt,
            onesided=onesided,
            qtilde=qtilde,
            onesideddiscovery=onesideddiscovery,
        )

        return pnull, palt

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, onesided, onesideddiscovery, qtilde):
        qalt = self.qalt(poinull, poialt, onesided=onesided, onesideddiscovery=onesideddiscovery)
        qalt = np.where(qalt < 0, 0, qalt)

        expected_pvalues = []
        for ns in nsigma:
            p_clsb = self.pnull(
                qobs=qalt,
                qalt=None,
                onesided=onesided,
                qtilde=qtilde,
                onesideddiscovery=onesideddiscovery,
                nsigma=ns,
            )
            if CLs:
                p_clb = norm.cdf(ns)
                p_cls = p_clsb / p_clb
                expected_pvalues.append(np.where(p_cls < 0, 0, p_cls))
            else:
                expected_pvalues.append(np.where(p_clsb < 0, 0, p_clsb))

        return expected_pvalues
