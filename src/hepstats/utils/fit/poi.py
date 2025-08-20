# Licensed under a 3-clause BSD style license, see LICENSE
"""
Module defining the parameter of interest classes, currently includes:

* **POIarray**
* **POI**
"""

from __future__ import annotations

from collections.abc import Collection
import numpy as np

from .. import api

class POIarray:
    """
    Class for parameters of interest with multiple values:
    """
    def __init__(
        self, 
        key : api.ParameterKey, 
        values: Collection[np.ndarray]
    ):
        
        """
        Args:
            parameter: the parameter of interest
            values: values of the parameter of interest

        Raises:
            ValueError: if is_valid_parameter(parameter) returns False
            TypeError: if parameter is not an iterable

        Example with `zfit`:
            >>> Nsig = zfit.Parameter("Nsig")
            >>> poi = POIarray(Nsig, value=np.linspace(0,10,10))
        """

        # if not isinstance(key, api.ParameterPathLike):
        #     msg = f"{param_spec} is not a valid parameter specification!"
        #     raise ValueError(msg)

        if not isinstance(values, Collection):
            msg = "A list/array of values of the POI is required."
            raise TypeError(msg)

        self.param_key = key
        self._values = np.array(values, dtype=np.float64)
        self._ndim = 1
        self._array_valued = isinstance(values[0], Collection)

    @property
    def values(self):
        """
        Returns the values of the **POIarray**.
        """
        return self._values

    def __repr__(self):
        return f"POIarray('{self.param_key.__repr__()}', values={self.values})"

    def __getitem__(self, i):
        """
        Get the i-th element the array of values of the **POIarray**.
        """
        return POI(self.param_key, self.values[i])

    def __iter__(self):
        for v in self.values:
            yield POI(self.param_key, v)

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        if not isinstance(other, POIarray):
            return NotImplementedError

        if len(self) != len(other):
            return False

        if not np.shape(self.values) == np.shape(other.values):
            return False
            
        values_equal = self.values == other.values
        name_equal = self.param_key == other.param_key
        return values_equal.all() and name_equal

    def __hash__(self):
        return hash((self.param_key, self.values.tostring()))

    # @property
    # def ndim(self):
    #     """
    #     Returns the number of dimension of the **POIarray**.
    #     """
    #     return self._ndim

    @property
    def shape(self):
        """
        Returns the shape of the **POIarray**.
        """
        return (len(self._values),)

    def append(self, values: np.ndarray | Collection[np.ndarray]):
        """
        Append values in the **POIarray**.

        Args:
            values: values to append
        """
        # if not isinstance(values, Collection):
        #     values = [values]
        # values = np.concatenate([self.values, values])
        # return POIarray(param_spec=self.param_path, values=values)
        
        # if not isinstance(values, Collection):
        #     values = [values]
        # if not isinstance(values, Collection) or (self._array_valued and not isinstance(values[0], Collection)):
        #     values = [values]
        # self._values.extend(values)

        values = np.array(values)
        if values.ndim == self.ndim - 1:
            values = np.expand_dims(values, axis=0)
        elif values.ndim != self.ndim:
            raise ValueError("Values to append must be the same shape as POIarray or one fewer dimension")

        self._values = np.concatenate((self._values, values), axis=0)


class POI(POIarray):
    """
    Class for single value parameter of interest:
    """

    def __init__(self, key : api.ParameterKey, value: int | float):
        """
        Args:
            parameter: the parameter of interest
            values: value of the parameter of interest

        Raises:
            TypeError: if value is an iterable

        Example with `zfit`:
            >>> Nsig = zfit.Parameter("Nsig")
            >>> poi = POI(Nsig, value=0)
        """
        # if isinstance(value, Collection):
        #     msg = "A single value for the POI is required."
        #     raise TypeError(msg)

        super().__init__(key=key, values=[value])
        self._value = value

    @property
    def value(self):
        """
        Returns the value of the **POI**.
        """
        return self._value

    def __iter__(self):
        yield self

    def append(self, values: np.ndarray | Collection[np.ndarray]):
        raise TypeError("POI cannot append additional values. Please use POIarray.")

    # def __eq__(self, other):
    #     if not isinstance(other, POI):
    #         return NotImplemented

    #     value_equal = self.value == other.value
    #     # name_equal = self.name == other.name
    #     return value_equal # and name_equal

    def __repr__(self):
        return f"POI('{self.param_key.__repr__()}', value={self.value})"

    # def __hash__(self):
    #     return hash((self.name, self.value))


def asarray(poi: POI) -> POIarray:
    """
    Transforms a **POI** instance into a **POIarray** instance.

    Args:
        poi: the parameter of interest.
    """
    return POIarray(param_key=poi.param_key, values=poi.values)