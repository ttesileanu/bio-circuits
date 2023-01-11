from typing import Any, Callable


class BaseCallback:
    def __init__(
        self, intent: str = "checkpoint", timing: str = "post", scope: str = "training"
    ):
        """Base class for callbacks.

        :param intent: callback's intent; can be
            "progress"      -- provide progress reports, such as a progress bar
            "checkpoint"    -- track changes in model parameters and assess need for
                               early termination
        :param timing: when callback should run with respect to the call to `*_step`;
            can be "pre" or "post" for "checkpoint", must be "post" for "progress"
        :param scope: which phase to run in; can be "training", "test", or "both"
        """
        self.intent = intent
        self.timing = timing
        self.scope = scope

    def initialize(self, trainer: Any):
        """Initialize the callback."""
        pass

    def finalize(self):
        """Finalize the callback."""
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """The callback function.

        The call signature depends on the attributes set during `__init__`.

        Callbacks with intent "progress" are called with a dictionary of items to add to
        the progress bar, similar to `tqdm.set_postfix`. They are not expected to return
        anything and any return value is ignored:

            progress_callback({"key1": value1, "key2": value2, ...})

        Progress callbacks should always have `timing` set to "post".

        Callbacks with intent "checkpoint" receive both the model and trainer as
        arguments, and are expected to return a `bool`. A true return value indicates
        that everything is OK and the run should continue; returning false instead ends
        the run.

            checkpoint_callback(model, trainer) -> bool
        """
        raise NotImplementedError("need to override __call__ in descendants")


class LambdaCallback(BaseCallback):
    """A callback based on a lambda function."""

    def __init__(
        self,
        fct: Callable,
        intent: str = "checkpoint",
        timing: str = "post",
        scope: str = "training",
    ):
        super().__init__(intent, timing, scope)
        self.fct = fct

    def __call__(self, *args, **kwargs):
        return self.fct(*args, **kwargs)
