from tqdm.auto import tqdm
from typing import Callable, Any, Dict

from ..util.callbacks import BaseCallback


class ProgressBar(BaseCallback):
    def __init__(self, backend: Callable = tqdm, scope: str = "training", **kwargs):
        """Create the progress bar.

        :param backend: back-end to use, with `tqdm`-like interface
        :param scope: when to run the progress bar ("training", "test", or "both")
        :param **kwargs: additional keyword arguments that will be passed to `backend`
            in `initialize()`
        """
        super().__init__("progress", "post", scope)
        self.backend = backend
        self.additional_kwargs = kwargs

        self.pbar = None

    def initialize(self, trainer: Any):
        """Initialize the progress bar.

        Uses `trainer.max_batches` to set the progress-bar total.
        """
        total = trainer.max_batches
        self.pbar = self.backend(total=total, **self.additional_kwargs)

    def finalize(self):
        self.pbar.close()

    def __call__(self, postfix: Dict[str, Any]):
        self.pbar.update(1)
        self.pbar.set_postfix(postfix, refresh=False)
