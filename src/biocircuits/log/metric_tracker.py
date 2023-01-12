from typing import Optional, Dict, Callable

from ..util.callbacks import BaseCallback


class MetricTracker(BaseCallback):
    """A callback that allows tracking of (scalar) metrics that are not being tracker by
    the model by default.

    This also allows using some of these metrics for progress information.

    A possible use case for this is to track the model's loss function in cases where
    the loss is not calculated during normal model execution (e.g., because the training
    dynamics is not directly based on the gradient of a loss).
    """

    def __init__(
        self,
        both: Optional[Dict[str, Callable]] = None,
        log: Optional[Dict[str, Callable]] = None,
        progress: Optional[Dict[str, Callable]] = None,
        frequency: int = 1,
        timing: str = "post",
        scope: str = "training",
    ):
        """Initialize the trainer.

        :param both: dictionary of items to use both with `model.log()` and
            `model.report_progress()`; the callable will be called with signature
            `callable(model, trainer)` and the output will be sent to the two model
            methods
        :param log: items to be sent only to `model.log()`
        :param progress: items to be sent only to `model.progress()`
        :param frequency: how often to perform the tracking
        :param timing: override timing of callback -- "pre" or "post"
        :param scope: override scope -- "training", "test", or "both"
        """
        super().__init__("checkpoint", timing, scope)
        self.both = both if both is not None else {}
        self.log = log if log is not None else {}
        self.progress = progress if progress is not None else {}
        self.frequency = frequency

    def __call__(self, model, trainer) -> bool:
        if trainer.batch_idx % self.frequency == 0:
            for key, callable in self.both.items():
                value = callable(model, trainer)
                model.log(key, value)
                model.report_progress(key, value)

            for key, callable in self.log.items():
                value = callable(model, trainer)
                model.log(key, value)

            for key, callable in self.progress.items():
                value = callable(model, trainer)
                model.report_progress(key, value)

        return True
