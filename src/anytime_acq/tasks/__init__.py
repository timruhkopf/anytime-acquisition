"""Pipeline registry: `task.name` in the config selects the function."""

from collections.abc import Callable

from omegaconf import DictConfig

from ..tracking.sink import Run
from .bo import bo
from .ei_readout import ei_readout
from .toy_regression import toy_regression

Pipeline = Callable[[DictConfig, Run], float | None]

TASKS: dict[str, Pipeline] = {
    "bo": bo,
    "ei_readout": ei_readout,
    "toy_regression": toy_regression,  # the template's example; its own tests (`experiment=debug`) use it
}
