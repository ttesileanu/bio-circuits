from copy import deepcopy
from typing import Union, Callable

from ..util.callbacks import BaseCallback


class ModelCheckpoint(BaseCallback):
    def __init__(
        self,
        timing: str = "post",
        scope: str = "training",
        criterion: Union[str, Callable] = "batch",
        frequency: int = 10,
    ):
        """Initialize the checkpoint callback.

        :param criterion: criterion used to decide if to make checkpoint; can be a
            callable with signature `criterion(model, trainer) -> bool`, or "batch", in
            which case see `frequency`
        :param frequency: when `criterion` is "batch", make a checkpoint whenever
            `trainer.batch_idx % frequency == 0`
        :param timing: whether to make the checkpoint before or after the step
        :param scope: whether to run in `training`, `test`, or `both`
        """
        super().__init__("checkpoint", timing, scope)
        self.checkpoints = []
        self.indices = {"batch_idx": [], "sample_idx": []}
        self.frequency = frequency

        if isinstance(criterion, str):
            if criterion != "batch":
                raise ValueError("unknown criterion")

            criterion = (
                lambda _, trainer, cb=self: trainer.batch_idx % (cb.frequency) == 0
            )

        self.criterion = criterion

    def __call__(self, model, trainer):
        if self.criterion(model, trainer):
            state_dict = deepcopy(model.state_dict())
            self.checkpoints.append(state_dict)
            self.indices["batch_idx"].append(trainer.batch_idx)
            self.indices["sample_idx"].append(trainer.sample_idx)
