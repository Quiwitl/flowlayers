"""flowlayers: fast probabilistic traversal of connectomes (Schlegel et al. 2021), on CPU or GPU."""
from .graph import Graph
from .model import TraversalModel, TraversalResult, linear_p, traverse

__version__ = "0.1.0"
__all__ = ["Graph", "TraversalModel", "TraversalResult", "linear_p", "traverse", "__version__"]
