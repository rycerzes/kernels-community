"""Vendored subset of quack 0.6.4 (https://github.com/Dao-AILab/quack), the modules ffpa-attn uses.

Importing `.dsl` installs quack's CuTe tensor indexing patch, exactly as upstream's `quack/__init__.py`
does; `copy_utils` indexes tensors with `...`, which CuTe does not support without it.
"""

from . import dsl  # noqa: F401
