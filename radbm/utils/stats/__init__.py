from .hypergeometric import (
    hypergeometric,
    superdupergeometric,
    superdupergeometric_expectations,
)

# Solution from https://stackoverflow.com/a/66996523
__all_exports = [
    hypergeometric,
    superdupergeometric,
    superdupergeometric_expectations,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]