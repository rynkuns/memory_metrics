# from .metrics import Metrics
from . import *

import importlib.metadata
__version__ = VERSION = importlib.metadata.version("memory_metrics")