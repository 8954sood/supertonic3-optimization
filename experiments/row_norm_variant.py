"""Load a benchmark engine with an explicitly verified optional kernel library."""
from unittest.mock import patch
from runtime import Engine


def engine_for_library(model, library=None):
    if not library:
        return Engine(model=model)
    from row_norm_ops import library_path
    verified = library_path(library)
    with patch('row_norm_ops.library_path', return_value=verified):
        return Engine(model=model)
