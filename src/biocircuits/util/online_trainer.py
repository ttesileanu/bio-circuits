import torch

from typing import Iterable, Any, Union, Optional

from .logger import Logger


class TrainingBatch:
    """Handle for one training batch.

    This helps train a network and can also help with logging. The batch object has
    `log`, `log_batch`, and `log_accumulated` methods that mirror those of a `Logger`.
    In addition to calling the corresponding `Logger` methods, these also log sample
    information using names obtained by adding `".sample"` to the `key`.

    The batch object also has methods `accumulate` and `calculate_accumulated` that
    simply call the corresponding `Logger` methods.

    Note that all logging-related methods return the *batch* as opposed to the logger.

    Attributes:
    :param data: batch data
    :param idx: batch index
    :param sample_idx: sample index for the start of the batch
    """

    def __init__(
        self,
        data: torch.Tensor,
        idx: int,
        sample_idx: int,
        iterator: "TrainingIterable",
    ):
        self.data = data
        n = len(self.data[0])
        for crt in self.data[1:]:
            assert len(crt) == n

        self.idx = idx
        self.sample_idx = sample_idx

        self._iterator = iterator
        self._trainer = self._iterator.trainer
        self._model = self._iterator.model
        self._logger = self._trainer.logger

    def training_step(self) -> Any:
        """Run one training step, returning the output from `model.forward()`."""
        if hasattr(self._model, "training_step"):
            out = self._model.training_step(self)
        else:
            out = self._model.forward(*self.data)
            self._model.backward(out)

        return out

    def every(self, step: int) -> bool:
        """Return true every `step` steps."""
        return self.idx % step == 0

    def count(self, total: int) -> bool:
        """Return true a total of `total` times.

        Including first and last batch.
        """
        # should be true when batch = floor(k * (n - 1) / (total - 1)) for integer k
        # this implies (batch * (total - 1)) % (n - 1) == 0 or > (n - total).
        if total == 1:
            return self.idx == 0
        else:
            n = len(self._iterator.loader)
            mod = (self.idx * (total - 1)) % (n - 1)
            return mod == 0 or mod > n - total

    def terminate(self):
        """Terminate the run early.

        Note that this does not stop the iteration instantly, but instead ends it the
        first time a new batch is requested. Put differently, the remaining of the `for`
        loop will still be run before it terminates.
        """
        self._iterator.terminating = True

    def log(
        self,
        key: Union[str, dict],
        value: Optional[Any] = None,
        index_prefix: Optional[str] = None,
    ) -> "TrainingBatch":
        """Log a value or set of values.

        In addition to calling `Logger.log()` for the given key-value combination, this
        also records the associated sample index in a key named `f"{key}.sample"`.

        :param key: key to log in, or dictionary of key-value pairs
        :param value: value to log; needed if `key` is not a `dict`
        :param index_prefix: key prefix to use for storing sample index; this is most
            useful if `key` is a `dict` as only one key is used for storing sample
            indices
        :return: self (NB: not the logger!).
        """
        self._logger.log(key, value)

        if index_prefix is not None:
            self._logger.log(f"{index_prefix}.sample", self.sample_idx)
        elif isinstance(key, str):
            self._logger.log(f"{key}.sample", self.sample_idx)
        else:
            for crt in key:
                self._logger.log(f"{crt}.sample", self.sample_idx)

        return self

    def log_batch(
        self,
        key: Union[str, dict],
        value: Optional[Any] = None,
        index_prefix: Optional[str] = None,
    ) -> "TrainingBatch":
        """Log a batch of values.

        In addition to calling `Logger.log()` for the given key-value combination, this
        also records the associated sample indices in a key named `f"{key}.sample"`.

        Note that the batch size is based on `self.data`, not on the shape of the logged
        values! It is the user's responsibility to make sure that these match.

        :param key: key to log in, or dictionary of key-value pairs
        :param value: value to log; needed if `key` is not a `dict`
        :param index_prefix: key prefix to use for storing sample index; this is most
            useful if `key` is a `dict` as only one key is used for storing sample
            indices
        :return: self (NB: not the logger!).
        """
        self._logger.log(key, value)

        samples = self.sample_idx + torch.arange(len(self))
        if index_prefix is not None:
            self._logger.log(f"{index_prefix}.sample", self.sample_idx)
        elif isinstance(key, str):
            self._logger.log_batch(f"{key}.sample", samples)
        else:
            for crt in key:
                self._logger.log_batch(f"{crt}.sample", samples)
        return self

    def accumulate(
        self, key: Union[str, dict], value: Optional[Any] = None
    ) -> "TrainingBatch":
        """Accumulate values to log later.

        This simply calls `Logger.accumulate()`.
        """
        self._logger.accumulate(key, value)
        return self

    def calculate_accumulated(self, key: str) -> Any:
        """Average all accumulated values for a given key.

        This simply calls `Logger.calculate_accumulated()`.
        """
        return self._logger.calculate_accumulated(key)

    def log_accumulated(self, index_prefix: Optional[str] = None) -> "TrainingBatch":
        """Store accumulated values.

        In addition to calling `Logger.log()` for all accumulated keys, this also
        records the associated sample indices in keys named `f"{key}.sample"`.

        :param index_prefix: key prefix to use for storing sample index; this is most
            useful if `key` is a `dict` as only one key is used for storing sample
            indices
        :return: self (NB: not the logger!).
        """
        keys = list(self._logger._accumulator.keys())
        self._logger.log_accumulated()

        if index_prefix is not None:
            self._logger.log(f"{index_prefix}.sample", self.sample_idx)
        else:
            for key in keys:
                self._logger.log(f"{key}.sample", self.sample_idx)

        return self

    def __len__(self) -> int:
        """Number of samples in batch."""
        return len(self.data[0])

    def __repr__(self) -> str:
        s = (
            f"TrainingBatch("
            f"data={self.data}, "
            f"idx={self.idx}, "
            f"sample_idx={self.sample_idx}"
            f")"
        )
        return s


class TrainingIterable:
    """Iterable returned by calling an OnlineTrainer, as well as corresponding iterator.

    Iterating through this yields `TrainingBatch`es. At the end of iteration, the
    `OnlineTrainer`'s `Logger`'s `finalize` method is called to prepare the results for
    easy access.
    """

    def __init__(
        self, trainer: "OnlineTrainer", model: torch.nn.Module, loader: Iterable
    ):
        self.trainer = trainer
        self.model = model
        self.loader = loader

        self.terminating = False

        self._it = None
        self._i = 0
        self._sample = 0

    def __iter__(self) -> "TrainingIterable":
        self.terminating = False

        self._i = 0
        self._sample = 0
        self._it = iter(self.loader)
        return self

    def __next__(self) -> TrainingBatch:
        try:
            if self.terminating:
                raise StopIteration

            data = next(self._it)

            batch = TrainingBatch(
                data=data, idx=self._i, sample_idx=self._sample, iterator=self
            )
            self._i += 1
            self._sample += len(batch)

            return batch

        except StopIteration:
            # ensure logger coalesces history at the end of the iteration
            self.trainer.logger.finalize()
            raise StopIteration

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        s = (
            f"TrainingIterable("
            f"trainer={repr(self.trainer)}, "
            f"model={repr(self.model)}, "
            f"loader={repr(self.loader)}, "
            f"terminating={self.terminating}, "
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"TrainingIterable("
            f"trainer={str(self.trainer)}, "
            f"model={repr(self.model)}, "
            f"loader={str(self.loader)}, "
            f")"
        )
        return s


class OnlineTrainer:
    """Manager for online training sections.

    Calling an `OnlineTrainer` object returns a `TrainingIterable`. Iterating through
    that iterable yields `TrainingBatch` objects, which can be used to train the network
    and report values to a `Logger`.

    Attributes
    :param logger: `Logger` object used for keeping track of reported tensors; it is
        generally best to use `TrainingBatch.log` and related functions for logging, and
        reserve `logger` for reading out the logged data
    """

    def __init__(self):
        self.logger = Logger()

    def __call__(self, model: torch.nn.Module, loader: Iterable) -> TrainingIterable:
        return TrainingIterable(self, model, loader)

    def __repr__(self) -> str:
        s = f"OnlineTrainer(logger={repr(self.logger)})"
        return s
