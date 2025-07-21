from collections import OrderedDict
import scipy.optimize as opt
import numpy as np

class Parameter:
    def __init__(self, name, value=None, lower=None, upper=None, floating=True):
        assert value > lower and value < upper, "Value not within bounds"
        
        self._name = name
        self._value = value
        self._lower = lower
        self._upper = upper
        self._floating = floating

    def value(self):
        return self._value

    def set_value(self, value):
        assert self._upper is None or value <= self._upper, "Value is too high; {}={}".format(self._name, value)
        assert self._lower is None or value >= self._lower, "Value is too low; {}={}".format(self._name, value)
        self._value = value
        return _TemporaryValueSetter(self, value)

    @property
    def name(self):
        return self._name

    @property
    def upper(self):
        return self._upper

    @property
    def lower(self):
        return self._lower

    @property
    def floating(self):
        return self._floating

    @floating.setter
    def floating(self, state):
        self._floating = state

    def __hash__(self):
        return hash(self._name)

class _TemporaryValueSetter:
    def __init__(self, param, new_value):
        self.param = param
        self.new_value = new_value
        self.old_value = param.value()

    def __enter__(self):
        self.param._value = self.new_value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.param._value = self.old_value
        return False  # propagate exceptions

class NegativeLogLikelihood:
    def __init__(self, params_map, func):
        self._model = []
        self._data = []
        self._fit_range = [None]
        self._constraints = []
        self._errordef = 0.5

        self._params_map = params_map
        self._func = func

        self._func_input_size = 0
        for par in self._params_map:
            self._func_input_size += len(self._params_map[par])
        
    def from_zfit(nll):
        params_map = OrderedDict()
        for ipar, zfit_par in enumerate(nll.get_params()):
            par = Parameter(zfit_par.name, zfit_par.value(), zfit_par.lower, zfit_par.upper, zfit_par.floating)
            params_map[par] = [ipar]
        
        return NegativeLogLikelihood(params_map, lambda x: nll(x))

    def __call__(self, param_values=None):
        if param_values is not None:
            assert len(param_values) == len(self._params_map), "Input size does no match number of parameters"
            for ipar, par in enumerate(self._params_map):
                par.set_value(param_values[ipar])

        func_input = np.array([None] * self._func_input_size)
        for par in self._params_map:
            for idx in self._params_map[par]:
                assert func_input[idx] is None, "Index collision in params map"
                func_input[idx] = par.value()
            
        return self._func(func_input)

    def __add__(self, other_nll):
        assert type(other_nll) == NegativeLogLikelihood, "Adding invalid class"
        
        new_params_map = self._params_map.copy()
        for par in other_nll._params_map:
            if par in new_params_map:
                new_params_map[par] += [idx + self._func_input_size for idx in other_nll._params_map]
            else:
                new_params_map[par] = [idx + self._func_input_size for idx in other_nll._params_map]

        new_func = lambda x: self._func(x[:self._func_input_size]) + other_nll._func(x[self._func_input_size:])
        return NegativeLogLikelihood(new_params_map, new_func)

    def value(self, param_values=None):
        return self(param_values)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def fit_range(self):
        return self._fit_range

    @property
    def constraints(self):
        return self._constraints

    @property
    def errordef(self):
        return self._errordef

    def get_params(self):
        return list(self._params_map)

class Minimum:
    def __init__(self, fmin, params, hess=None):
        self._fmin = fmin
        self._params = params
        self._hess = hess

    @property
    def fmin(self):
        return self._fmin

    @property
    def params(self):
        return self._params

    @property
    def hess(self):
        return self._hess

class Minimizer:
    def __init__(self):
        pass

    def minimize(self, loss):
        floating_idx = []
        floating_bounds = []
        input_arr = np.empty(len(loss.get_params()))

        for ipar, par in enumerate(loss.get_params()):
            if par.floating:
                floating_idx.append(ipar)
                floating_bounds.append((par.lower, par.upper))
            input_arr[ipar] = par.value()
        floating_idx = np.array(floating_idx)

        def target_func(floating_vals):
            input_arr[floating_idx] = floating_vals
            return loss(input_arr)

        min_result = opt.minimize(target_func, input_arr[floating_idx], bounds=floating_bounds)
        assert min_result.success, "Minimizer did not converge"

        input_arr[floating_idx] = min_result.x
        return Minimum(min_result.fun, {par : {"value": val} for par, val in zip(loss.get_params(), input_arr)})