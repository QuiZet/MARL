# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Copyright 2023 RTPlayground Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Decorator definitions
"""

# https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically

import functools
import inspect
import warnings

string_types = (type(b''), type(u''))

from typing import Callable, List
from MARL.utils_log import comms

def decorate_all(decorators: List[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class

    Args:
        decorators: list of decorators to add to all functions in the class
    """

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != "__init__":
                for decorator in decorators:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def check_profiler_enabled(func: Callable) -> Callable:
    """Decorator: check if profiler is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.enable_profiler:
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_viewer_enabled(func: Callable) -> Callable:
    """Decorator: check if viewer is enabled and only run on main process"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.is_viewer_enabled() and comms.is_main_process():
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_eval_enabled(func: Callable) -> Callable:
    """Decorator: check if evaluation step is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.is_wandb_enabled() or self.config.is_tensorboard_enabled():
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if comms.is_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper

string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))

def run_once(f):
    """ Run the function only one time.
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper