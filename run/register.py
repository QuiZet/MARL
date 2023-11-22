import time
import os
import importlib

TRAINER_MAP = {}

def get_trainer(key):
    return TRAINER_MAP[key]

#def register_trainer(key, cls):
#    TRAINER_MAP[key] = cls


def register_trainer(func):
    """Registers a function.

    Args:
        func: The function to register.

    Returns:
        The registered function.

    Example:

    newfile.py
    from register import register_trainer
    @register_trainer
    def myfunc(*args, **kwargs) -> None:

    main.py
    import register
    import newfile

    # start trainer
    register.get_trainer('myfunc')(...)          
    """
    TRAINER_MAP[func.__name__] = func
    #TRAINER_MAP[func.__name__] = []
    #TRAINER_MAP[func.__name__].append(func)

    return func
  
  
  
class FunctionRegistry:
  """A class for registering functions.

  Attributes:
    functions: A list of registered functions.
  """

  def __init__(self):
    self.functions = []

  def register(self, func):
    """Registers a function.

    Args:
      func: The function to register.
    """

    self.functions.append(func)

  def get_functions(self):
    """Returns a list of registered functions.

    Returns:
      A list of registered functions.
    """

    return self.functions

# Create the function registry.
#registry = FunctionRegistry()

# Register the function.
#registry.register(my_function)

# Get the registered functions.
#functions = registry.get_functions()

# Print the names of the registered functions.
#for function in functions:
#  print(function.__name__)    