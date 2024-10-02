from typing import Callable


def is_documented_by(original:Callable) -> Callable:
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper
