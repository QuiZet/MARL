from typing import Any

# https://discuss.python.org/t/how-to-convert-dictionary-into-local-variables-inside-the-function/16420/4
class AttrDict(dict):
    """Provide dictionary with items accessible as object attributes."""
    def __getattr__(self, attr: str) -> Any:
        try:
            return self[attr]
        except KeyError as exception:
            raise AttributeError(f'AttrDict has no key {attr!r}') from exception

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value

    def __delattr__(self, attr: str) -> Any:
        try:
            del self[attr]
        except KeyError as exception:
            raise AttributeError(f'AttrDict has no key {attr!r}') from exception
