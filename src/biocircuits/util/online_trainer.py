import torch

from collections import defaultdict
from typing import Iterable, Union, Optional, List, Callable, Dict, Optional

from ..log.logger import Logger
from ..util.callbacks import BaseCallback
from ..arch.base import BaseOnlineModel


class OnlineTrainer:
    def __init__(
        self,
        callbacks: Optional[List[Union[Callable, BaseCallback]]] = None,
        logger: Optional[Logger] = None,
    ):
        """Initialize the trainer.

        :param callbacks: list of callback functions; see `BaseCallback`
        :param logger: logger object; if `None`, a new `Logger` object will be created;
            this logger will be used by the model unless overridden by `model.logger`
        """
        self.callbacks = callbacks if callbacks is not None else []
        self.logger = logger if logger is not None else Logger()
        self.max_batches = None
        self.batch_idx = None
        self.sample_idx = None

    def fit(
        self, model: BaseOnlineModel, loader: Iterable
    ) -> Optional[List[torch.Tensor]]:
        """Train a model on a dataset.

        :param model: model to train
        :param loader: iterable used to generate training batches
        :return: a tensor containing all outputs from `training_step`, or `None` if
            there is no output
        """
        callbacks = self._callbacks_by_scope("training")
        assert len(callbacks["pre_progress"]) == 0

        self.max_batches = len(loader)
        self.batch_idx = 0
        self.sample_idx = 0

        self.logger.initialize()
        for callback_list in callbacks.values():
            for callback in callback_list:
                callback.initialize(self)

        outputs = None

        model.trainer = self
        for batch in loader:
            stopping = False
            for callback in callbacks["pre_checkpoint"]:
                if not callback(model, self):
                    stopping = True
                    break
            if stopping:
                break

            crt_output = model.training_step(batch)
            if crt_output is not None:
                if outputs is None:
                    outputs = []
                outputs.append(crt_output)
            elif outputs is not None:
                outputs.append(crt_output)

            progress_dict = model.for_progress
            for callback in callbacks["post_progress"]:
                callback(progress_dict)

            stopping = False
            for callback in callbacks["post_checkpoint"]:
                if not callback(model, self):
                    stopping = True
                    break
            if stopping:
                break

            self.batch_idx += 1
            self.sample_idx += len(batch)

        self.logger.finalize()

        for callback_list in callbacks.values():
            for callback in callback_list:
                callback.finalize()

        return outputs

    def predict(
        self, model: BaseOnlineModel, loader: Iterable
    ) -> Optional[List[torch.Tensor]]:
        """Run inference using the model on the dataset.

        :param model: model to use for inference
        :param loader: iterable used to generate test batches
        :return: a tensor containing all outputs from `test_step`, or `None` if there is
            no output
        """
        callbacks = self._callbacks_by_scope("test")
        assert len(callbacks["pre_progress"]) == 0

        for callback_list in callbacks.values():
            for callback in callback_list:
                callback.initialize(self)

        outputs = None

        model.trainer = self
        for batch in loader:
            for callback in callbacks["pre_checkpoint"]:
                callback(model)

            stopping = False
            for callback in callbacks["pre_monitor"]:
                if not callback(model, self):
                    stopping = True
                    break
            if stopping:
                break

            crt_output = model.test_step(batch)
            if crt_output is not None:
                if outputs is None:
                    outputs = []
                outputs.append(crt_output)
            elif outputs is not None:
                outputs.append(crt_output)

            progress_dict = model.for_progress
            for callback in callbacks["post_progress"]:
                callback(progress_dict)
            for callback in callbacks["post_checkpoint"]:
                callback(model)

            stopping = False
            for callback in callbacks["post_monitor"]:
                if not callback(model, self):
                    stopping = True
                    break
            if stopping:
                break

        for callback_list in callbacks.values():
            for callback in callback_list:
                callback.finalize()

        return outputs

    def _callbacks_by_scope(self, scope: str) -> Dict[str, BaseCallback]:
        """Get the callbacks with the given scope, split depending on intent and when
        they should be called.

        Specifically, this returns a dictionary with keys of the form "pre_checkpoint",
        "post_checkpoint", etc., identifying the type of callback and the timing of the
        call. Each key maps to a list of callback functions.
        """
        d = defaultdict(list)
        for callback in self.callbacks:
            assert callback.intent in ["progress", "checkpoint"]
            if callback.scope in [scope, "both"]:
                key = f"{callback.timing}_{callback.intent}"
                d[key].append(callback)

        return d

    def __repr__(self) -> str:
        s = (
            f"OnlineTrainer("
            f"callbacks={repr(self.callbacks)}, "
            f"logger={repr(self.logger)}"
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"OnlineTrainer("
            f"callbacks={str(self.callbacks)}, "
            f"logger={str(self.logger)}"
            f")"
        )
        return s
