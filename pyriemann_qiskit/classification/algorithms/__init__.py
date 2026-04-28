"""Quantum classification algorithms."""

import logging

from .continuous_qioce_classifier import ContinuousQIOCEClassifier
from .cp_mdm import CpMDM
from .nearest_convex_hull import NearestConvexHull
from .quantum_state_discriminator import QuantumStateDiscriminator

logging.basicConfig(level=logging.WARNING)

__all__ = [
    "ContinuousQIOCEClassifier",
    "CpMDM",
    "NearestConvexHull",
    "QuantumStateDiscriminator",
]
