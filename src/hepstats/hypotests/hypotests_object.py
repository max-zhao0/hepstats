from __future__ import annotations

import warnings

import numpy as np

# from ..utils.pytree import pt
# from ..utils.fit import get_nevents, set_values_once
# # from ..utils.fit.api import ModelLike, LossLike, MinimizerLike, MinimumLike, ParameterLike, convert_params
# from ..utils.fit
# # from .parameters import POI
# from ..utils.fit import minimizers
from ..utils import pt, api, minimizers
from ..utils import get_nevents, set_values_once

class HypotestsObject:
    """Base object in `hepstats.hypotests` to manipulate a loss function and a minimizer.

    Args:
        input: loss or fit result
        minimizer: minimizer to use to find the minimum of the loss function
    """

    def __init__(self, 
        loss : api.LossLike, 
        params : pt.PyTree[api.ParameterLike], 
        *loss_args,
        data : api.Data = None,
        minimizer : api.MinimizerLike = None, 
        **kwargs
    ):
        
        super().__init__(**kwargs)
        if not isinstance(loss, api.LossLike):
            raise ValueError("loss is not LossLike")

        self._loss = loss
        self._bestfit = None 
        self._parameters = api.convert_params(params)
        self._user_params = params
        self._data = data
        self._loss_args = loss_args

        if minimizer is None:
            self._minimizer = minimizers.IMinuit()
        elif not isinstance(minimizer, api.MinimizerLike):
            msg = f"{minimizer} is not a valid minimizer !"
            raise ValueError(msg)
        else:
            self._minimizer = minimizer

    @property
    def loss(self) -> LossLike:
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
            minimum = self.minimizer.minimize(loss=self.loss, params=self.parameters, data=self.data, *self._loss_args)
            self._bestfit = minimum
        return self._bestfit

    @bestfit.setter
    def bestfit(self, value : MinimumLike):
        """
        Set the best fit values  of the model parameters.

        Args:
            value: fit result
        """
        # CHECK PARAM TREEDEF
        if not isinstance(value, MinimumLike):
            msg = f"{value} is not a valid fit result!"
            raise ValueError(msg)
        self._bestfit = value

    # @property
    # def model(self):
    #     """
    #     Returns the model.
    #     """
    #     return self.loss.model

    # @property
    # def constraints(self):
    #     """
    #     Returns the constraints on the loss / likehood function.
    #     """
    #     return self.loss.constraints

    # def get_parameter(self, name: str):
    #     """
    #     Returns the parameter in loss function with given input name.

    #     Args:
    #         name: name of the parameter to return
    #     """
    #     return self._parameters[name]

    @property
    def parameters(self):
        """
        Returns the list of free parameters in loss / likelihood function in the form of InternalParameter.
        """
        return self._parameters

    # def set_params_to_bestfit(self):
    #     """
    #     Set the values of the parameters in the models to the best fit values
    #     """
    #     set_values_once(self.parameters, self.bestfit)

    # def lossbuilder(self, model, data, weights=None, oldloss=None):
    #     """Method to build a new loss function.

    #     Args:
    #         * **model** (List): The model or models to evaluate the data on
    #         * **data** (List): Data to use
    #         * **weights** (optional, List): the data weights
    #         * **oldloss**: Previous loss that has data, models, type

    #     Example with `zfit`:
    #         >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
    #         >>> mean = zfit.Parameter("mu", 1.2)
    #         >>> sigma = zfit.Parameter("sigma", 0.1)
    #         >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    #         >>> loss = calc.lossbuilder(model, data)

    #     Returns:
    #         Loss function

    #     """

    #     if oldloss is None:
    #         oldloss = self.loss
    #     assert all(is_valid_pdf(m) for m in model)
    #     assert all(is_valid_data(d) for d in data)

    #     msg = "{0} must have the same number of components as {1}"
    #     if len(data) != len(self.data):
    #         raise ValueError(msg.format("data", "`self.data"))
    #     if len(model) != len(self.model):
    #         raise ValueError(msg.format("model", "`self.model"))
    #     if weights is not None and len(weights) != len(self.data):
    #         raise ValueError(msg.format("weights", "`self.data`"))

    #     if weights is not None:
    #         for d, w in zip(data, weights):
    #             d = d.with_weights(w)

    #     if hasattr(oldloss, "create_new"):
    #         loss = oldloss.create_new(model=model, data=data, constraints=self.constraints)
    #     else:
    #         warnings.warn(
    #             "A loss should have a `create_new` method. If you are using zfit, please make sure to"
    #             "upgrade to >= 0.6.4",
    #             FutureWarning,
    #             stacklevel=2,
    #         )
    #         loss = type(oldloss)(model=model, data=data)
    #         loss.add_constraints(self.constraints)

    #     return loss


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
