"""Load a benchmark engine with an explicitly verified optional kernel library."""
from unittest.mock import patch
from runtime import Engine


def engine_for_library(model, library=None):
    if not library:
        return Engine(model=model)
    from row_norm_ops import library_path
    original = library_path
    with patch('row_norm_ops.library_path', side_effect=lambda **kwargs: original(library, **kwargs)):
        return Engine(model=model)
