from contextlib import ExitStack

import numpy as np

class Parameter:
    def __init__(self, value, floating, upper, lower):
        self.value = value
        self.floating = floating
        self.upper = upper
        self.lower = lower
    
    def from_zfit(zfit_param):
        return Parameter(float(zfit_param.value()), zfit_param.floating, float(zfit_param.upper), float(zfit_param.lower))

    def concat(parameters, floating):
        return Parameter(
            np.array([p.value for p in parameters]),
            floating,
            np.array([p.upper for p in parameters]),
            np.array([p.lower for p in parameters])
        )

class ExtendedUnbinnedModel:
    def from_zfit(zfit_model, unwrap_params=lambda x: x):
        model = ExtendedUnbinnedModel()
        
        model.backend = "zfit"
        model.zfit_model = zfit_model
        model.lower = zfit_model.norm.limits[0][0,0]
        model.upper = zfit_model.norm.limits[1][0,0]
        model.unwrap_params = unwrap_params

        model.pdf_vectorized = True
        return model

    def pdf(self, x, params):
        if self.backend == "zfit":
            return self.zfit_model.pdf(x, params=self.unwrap_params(params))
        else:
            raise NotImplementedError

    def get_yield(self, params):
        params = self.unwrap_params(params)
        if self.backend == "zfit":
            with ExitStack() as stack:
                for p in self.zfit_model.get_params():
                    stack.enter_context(p.set_value(params[p.name]))
                tot_yield = self.zfit_model.get_yield().value()
            return tot_yield
        else:
            raise NotImplementedError

class BinnedModel:
    def from_zfit(zfit_model, unwrap_params=lambda x: x):
        model = BinnedModel()

        model.backend = "zfit"
        model.zfit_model = zfit_model
        model.unwrap_params = unwrap_params
        
        return model

    def expected_histogram(self, params):
        params = self.unwrap_params(params)
        if self.backend == "zfit":
            with ExitStack() as stack:
                for p in self.zfit_model.get_params():
                    stack.enter_context(p.set_value(params[p.name]))
                hist = self.zfit_model.counts().numpy()
            return hist
        else:
            raise NotImplementedError

class BinnedNLL:
    def from_zfit(zfit_nll, unwrap_params=lambda x: x):
        nll = BinnedNLL()
        nll.backend = "zfit"
        nll.base_nll = zfit_nll
        nll.unwrap_params = unwrap_params
        nll.cache = []
        return nll

    def __call__(self, params, data):
        params = self.unwrap_params(params)
        if self.backend == "zfit":
            nll = None
            for pair in self.cache:
                if np.all(pair[0] == data):
                    nll = pair[1]
                    break
                    
            if nll is None:
                import zfit
                
                zfit_data = zfit.data.BinnedData.from_tensor(self.base_nll.model[0].space, data)
                nll = self.base_nll.create_new(data=zfit_data)
                self.cache.append((data, nll))
                

            with ExitStack() as stack:
                for p in nll.get_params():
                    stack.enter_context(p.set_value(params[p.name]))
                nll_val = nll.value()
            return nll_val
        else:
            raise NotImplementedError

class ExtendedUnbinnedNLL:
    def __init__(self, model):
        self.model = model

    def from_zfit(zfit_model, unwrap_params=None):
        return ExtendedUnbinnedNLL(ExtendedUnbinnedModel.from_zfit(zfit_model, unwrap_params))

    def yield_term(self, params):
        return self.model.get_yield(params)

    def pdf_term(self, params, data):
        return - len(data) * np.log(self.model.get_yield(params)) - np.sum(np.log(self.model.pdf(data, params)))