from .membership import MultiBernoulliMembershipMatch

__all_exports = [
    MultiBernoulliMembershipMatch,
]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]