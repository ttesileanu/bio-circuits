from typing import Any


class BaseCallback:
    def __init__(
        self, intent: str = "checkpoint", timing: str = "post", scope: str = "training"
    ):
        """Base class for callbacks.

        :param intent: callback's intent; can be
            "progress"      -- provide progress reports, such as a progress bar
            "checkpoint"    -- track changes in model parameters
            "monitor"       -- assess the need for early termination
        :param timing: when callback should run with respect to the call to `*_step`;
            can be "pre" or "post"
        :param scope: which phase to run in; can be "training", "test", or "both"
        """
        self.intent = intent
        self.timing = timing
        self.scope = scope

    def __call__(*args, **kwargs) -> Any:
        """The callback function.

        The call signature depends on the attributes set during `__init__`.

        Callbacks with intent "progress" are called with a dictionary of items to add to
        the progress bar, similar to `tqdm.set_postfix`. They are not expected to return
        anything and any return value is ignored:

            progress_callback({"key1": value1, "key2": value2, ...})

        Progress callbacks should always have `timing` set to "post".

        Callbacks with intent "checkpoint" receive the model as argument and are
        expected to return nothing (any return value is ignored):

            checkpoint_callback(model)

        Callbacks with intent "monitor" receive both the model and trainer as argument,
        and are expected to return a `bool`. A true return value indicates that
        everything is OK and the run should continue; returning false instead ends the
        run.

            monitor_callback(model, trainer) -> bool
        """
        raise NotImplementedError("need to override __call__ in descendants")
