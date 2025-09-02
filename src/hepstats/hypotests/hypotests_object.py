from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from ..utils import api, minimizers

class HypotestsObject:
    """Base object in `hepstats.hypotests` to manipulate a loss function and a minimizer.

    Args:
        input: loss or fit result
        minimizer: minimizer to use to find the minimum of the loss function
    """

    def __init__(self, 
        nll : api.LossLike | api.NegativeLogLikelihoodlike, 
        params : dict[api.ParameterKey, api.ParameterLike], 
        *loss_args,
        data : dict[api.DataKey, ArrayLike] = None,
        models : dict[api.DataKey, api.ModelLike] = None,
        minimizer : api.MinimizerLike = None, 
        **kwargs
    ):
        
        super().__init__(**kwargs)

        if not (isinstance(nll, api.LossLike) or isinstance(nll, api.NegativeLogLikelihoodLike)):
            raise ValueError("{} is not LossLike nor NegativeLogLikelihoodLike".format(nll))
            
        params = self.assert_dictlike(params)
        for k in params:
            if not isinstance(params[k], api.ParameterLike):
                raise ValueError("{} is not ParameterLike".format(params[k]))

        if data is not None:
            data = self.assert_dictlike(data)

        if models is not None:
            models = self.assert_dictlike(models)
            if models.keys() != data.keys():
                raise ValueError("{} and {} keys do not match".format(data, params))
            for dk in models:
                if isinstance(models[dk], api.BinnedModelLike):
                    assert False
                elif isinstance(models[dk], api.UnbinnedModelLike):
                    if not (np.shape(models[dk].lower) == np.shape(models[dk].upper) and np.ndim(models[dk].lower) <= 1):
                        raise ValueError("Model lower and upper must be one dimensional arrays of the same length")
                else:
                    raise ValueError("{} is not a valid model. It must be BinnedModelLike or UnbinnedModelLike".format(models[dk]))

        self._nll = nll
        self._bestfit = None 
        self._parameters = {k : api.InternalParameter(params[k]) for k in params}
        self._data = data
        self._models = models
        self._loss_args = loss_args

        self._loss = self._nll if self._data is None else self.lossbuilder(self._nll, self._data)

        if minimizer is None:
            self._minimizer = minimizers.IMinuit()
        elif not isinstance(minimizer, api.MinimizerLike):
            msg = f"{minimizer} is not a valid minimizer!"
            raise ValueError(msg)
        else:
            self._minimizer = minimizer

    def assert_dictlike(self, obj):
        try:
            return dict(obj)
        except TypeError:
            raise ValueError("{} is not dictlike".format(obj)) 

    @property
    def nll(self) -> api.NegativeLogLikelihoodLike:
        return self._nll
    
    @property
    def loss(self) -> api.LossLike:
        """
        Returns the loss / likelihood function.
        """
        return self._loss

    @property
    def minimizer(self):
        """
        Returns the minimizer.
        """
        return self._minimizer

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = self.assert_dictlike(value)
        self._loss = self.lossbuilder(self.nll, value)

    @property
    def models(self):
        return self._models

    @property
    def loss_args(self):
        return self._loss_args

    @property
    def bestfit(self) -> MinimumLike:
        """
        Returns the best fit values of the model parameters.
        """
        # if getattr(self, "_bestfit", None):
        #     return self._bestfit
        if self._bestfit is None:
            minimum = self.minimizer.minimize(loss=self.loss, params=self.parameters, *self._loss_args)
            self._bestfit = minimum
        return self._bestfit

    @bestfit.setter
    def bestfit(self, value : MinimumLike):
        """
        Set the best fit values  of the model parameters.

        Args:
            value: fit result
        """
        if not isinstance(value, MinimumLike):
            msg = f"{value} is not a valid fit result!"
            raise ValueError(msg)
        self._bestfit = value

    @property
    def models(self):
        """
        Returns the model.
        """
        return self._models

    def get_parameter(self, param_key : api.ParameterKey):
        """
        Returns the parameter in loss function with given input name.

        Args:
            name: name of the parameter to return
        """
        return self._parameters[param_key]

    @property
    def parameters(self):
        """
        Returns the list of free parameters in loss / likelihood function in the form of InternalParameter.
        """
        return self._parameters

    def lossbuilder(self,
        nll : api.NegativeLogLikelihoodLike,
        data : dict[api.DataKey, ArrayLike],
        corrections : list = []
    ) -> api.LossLike:
        def loss(params, *loss_args):
            total_correction = 0
            for corr in corrections:
                total_correction += corr(params)
            return total_correction + nll(params, *loss_args, data=data)
        return loss

class ToysObject(HypotestsObject):
    """Base object in `hepstats.hypotests` to manipulate a loss function, a minimizer and sample a
    model (within the loss function) to do toy experiments.

        Args:
            input: loss or fit result
            minimizer: minimizer to use to find the minimum of the loss function
            sampler: function used to create sampler with models, number of events and floating parameters
            in the sample.
            sample: function used to get samples from the sampler.
    """

    def __init__(self, input, minimizer, sampler, sample):
        super().__init__(input, minimizer)
        self._toys = {}
        self._sampler = sampler
        self._sample = sample
        self._toys_loss = {}

    def sampler(self):
        """
        Create sampler with models.

        >>> sampler = calc.sampler()
        """
        self.set_params_to_bestfit()
        nevents = []
        for m, d in zip(self.loss.model, self.loss.data):
            nevents_data = get_nevents(d)
            if m.is_extended:
                nevents.append(np.random.poisson(lam=nevents_data))  # TODO: handle constraint yields correctly?
            else:
                nevents.append(nevents_data)

        return self._sampler(self.loss.model, nevents)

    def sample(self, sampler, ntoys, poi: POI, constraints=None):
        """
        Generator function of samples from the sampler for a given value of a parameter of interest. Returns a
        dictionnary of samples constraints in any.

        Args:
            sampler (list): generator of samples
            ntoys (int): number of samples to generate
            poi (POI):  in the sampler
            constraints (list, optional): list of constraints to sample

        Example with `zfit`:
            >>> mean = zfit.Parameter("mean")
            >>> sampler = calc.sampler()
            >>> sample = calc.sample(sampler, 1000, POI(mean, 1.2))

        Returns:
            dictionnary of sampled values of the constraints at each iteration
        """
        return self._sample(
            sampler,
            ntoys,
            parameter=poi.parameter,
            value=poi.value,
            constraints=constraints,
        )

    def toys_loss(self, parameter_name: str):
        """
        Construct a loss function constructed with a sampler for a given floating parameter

        Args:
            parameter_name: name floating parameter in the sampler
        Returns:
             Loss function

        Example with `zfit`:
            >>> loss = calc.toys_loss(zfit.Parameter("mean"))
        """
        if parameter_name not in self._toys_loss:
            parameter = self.get_parameter(parameter_name)
            sampler = self.sampler()
            self._toys_loss[parameter.name] = self.lossbuilder(self.model, sampler)
        return self._toys_loss[parameter_name]
