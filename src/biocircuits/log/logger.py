import torch
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Union, Dict, Optional


class Logger:
    """Logger for scalar values.

    You can access the stored values through `storage` (a list of dictionaries) or `df`
    (a Pandas dataframe).

    Attributes:
    :param stage: dictionary holding a temporary stash of stored values; use `commit()`
        to append to `storage`
    :param storage: list of dictionaries of permanently stored values
    :param df: stored values as a Pandas dataframe; this gets generated on the fly and
        is cached; the cache is invalidated every time `commit()` is called
    """

    def __init__(self):
        self.stage = defaultdict(list)
        self.storage = []
        self.index = 0
        self._df = None

    def log(
        self,
        key: Union[str, dict],
        value: Union[None, int, float, torch.Tensor, np.ndarray] = None,
    ) -> "Logger":
        """Log a value or set of values.

        This takes the `item()` of a single-element tensor, or selects the only element
        of an Numpy array. Non-scalar values are not allowed.

        Multiple values can be recorded at once by using a `dict` as first argument:
            logger.log({"foo": 2, "bar": 3})

        :param key: name assigned to stored value, or dictionary `{name: value}`
        :param value: value to store; needed unless `key` is a `dict`
        :return: `self`
        """
        if not isinstance(key, str):
            for k, v in key.items():
                self.log(k, v)
        else:
            self.stage[key].append(_convert(value))

        return self

    def commit(self):
        """Commit staged values to the storage area.

        This also invalidates the cached dataframe. Does nothing if the stage is empty.
        """
        if len(self.stage) > 0:
            self.storage.append(self._process_stage())
            self.stage.clear()
            self._df = None

    def step(self, index: Optional[int] = None):
        """Commit staged values and update index.

        :param index: value to set the index to after committing staged values; if not
            provided, increment current value
        """
        self.commit()
        if index is None:
            self.index += 1
        else:
            self.index = index

    def save(self):
        """Convert stored values to Pandas dataframe.

        There is little reason to call this directly, as it is called automatically
        when accessing `self.df`.
        """
        df = pd.DataFrame(self.storage)
        if len(df) > 0:
            df.set_index("index", inplace=True)
        self._df = df

    def _process_stage(self) -> Dict[str, Union[int, float]]:
        """Process the stage by aggregating over repeated values and appending index."""
        processed = {"index": self.index}
        for key, values in self.stage.items():
            if len(values) == 1:
                value = values[0]
            else:
                value = np.mean(values)

            processed[key] = value

        return processed

    @property
    def df(self):
        """Access the stored values as a Pandas dataframe."""
        if self._df is None:
            self.save()

        return self._df

    def initialize(self):
        """Initialize the logger.

        Currently this just empties the stage.
        """
        self.stage.clear()

    def finalize(self):
        """Finalize the logger.

        This calls `step()` to store anything that is left in the staging area and
        increment the index.
        """
        self.step()

    def __repr__(self) -> str:
        s = f"Logger()"

        return s


def _convert(value: Union[int, float, torch.Tensor, np.ndarray]) -> Union[int, float]:
    if torch.is_tensor(value):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.item()
    elif hasattr(value, "__len__"):
        raise ValueError("trying to log non-scalar; only scalars supported")
    else:
        return value
