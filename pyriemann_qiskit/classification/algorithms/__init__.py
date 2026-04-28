"""Quantum classification algorithms."""

import logging

from .cp_mdm import CpMDM
from .nearest_convex_hull import NearestConvexHull

logging.basicConfig(level=logging.WARNING)

__all__ = ["CpMDM", "NearestConvexHull"]
