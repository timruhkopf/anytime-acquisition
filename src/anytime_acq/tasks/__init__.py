"""Pipeline registry: `task.name` in the config selects the function."""

from collections.abc import Callable

from omegaconf import DictConfig

from ..tracking.sink import Run
from .api import api
from .api_decoder import api_decoder
from .api_gate import api_gate
from .api_probe import api_probe
from .bo import bo
from .toy_regression import toy_regression

Pipeline = Callable[[DictConfig, Run], float | None]

TASKS: dict[str, Pipeline] = {
    "api": api,
    "api_decoder": api_decoder,
    "api_gate": api_gate,
    "api_probe": api_probe,
    "bo": bo,
    "toy_regression": toy_regression,  # the template's example; its own tests (`experiment=debug`) use it
}
