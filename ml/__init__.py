"""
AstroGuard Machine Learning Module
===================================
Satellite collision prediction using LSTM neural networks.
"""

__version__ = "1.0.0"

from ml.utils import (
    tle_to_cartesian,
    calculate_distance,
    calculate_relative_velocity,
    risk_classification
)

from ml.preprocessing import OrbitDataPreprocessor
from ml.train import CollisionPredictor
from ml.predict import CollisionRiskPredictor

__all__ = [
    'OrbitDataPreprocessor',
    'CollisionPredictor',
    'CollisionRiskPredictor',
    'tle_to_cartesian',
    'calculate_distance',
    'calculate_relative_velocity',
    'risk_classification'
]
