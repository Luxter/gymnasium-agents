from inspect import getargvalues
from types import FrameType
from typing import Any


def get_function_parameters(frame: FrameType) -> dict[str, Any]:
    args, _, _, values = getargvalues(frame)
    params = {arg: values[arg] for arg in args}
    return params
