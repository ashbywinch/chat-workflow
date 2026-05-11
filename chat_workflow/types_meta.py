import inspect
import typing
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel


def resolve_return_type(
    raw_func: Callable[..., Any],
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any | None:
    """Resolve the actual return type of a function.

    Handles Self, TypeVar, and concrete BaseModel return types.
    """
    return_type = typing.get_type_hints(raw_func).get("return")

    if return_type is typing.Self:
        return resolve_self_type(raw_func, func)

    elif isinstance(return_type, TypeVar):
        return resolve_generic_type(raw_func, func, args, kwargs, return_type)

    else:
        return return_type


def resolve_generic_type(raw_func, func, args, kwargs, return_type):
    bound_type = return_type.__bound__
    if bound_type is None or not issubclass(bound_type, BaseModel):
        raise TypeError(
            f"Leaf function '{func.__name__}' TypeVar bound must be a Pydantic model, bound is {bound_type}"
        )
    actual_return_type: type[BaseModel] = return_type
    param_hints = typing.get_type_hints(raw_func)
    typevar_params = [
        name for name, annotation in param_hints.items() if name != "return" and annotation == return_type
    ]
    for param_name in typevar_params:
        if param_name in kwargs:
            actual_return_type = type(kwargs[param_name])
            break
        sig = inspect.signature(raw_func)
        param_names = list(sig.parameters.keys())
        if param_name in param_names:
            idx = param_names.index(param_name)
            if idx < len(args):
                actual_return_type = type(args[idx])
                break
    return actual_return_type


def resolve_self_type(raw_func, func):
    qualname = raw_func.__qualname__
    parts = qualname.rsplit(".", 1)
    if len(parts) < 2:
        raise TypeError(f"Leaf function '{func.__name__}' uses Self return type but is not a method on a class")
    class_name = parts[0]
    return raw_func.__globals__.get(class_name)
