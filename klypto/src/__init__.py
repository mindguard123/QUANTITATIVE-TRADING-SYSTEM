"""
Quantitative Trading System for Indian Markets
Author: Senior Quantitative Researcher
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from . import data_acquisition
from . import feature_engineering
from . import hmm_regime
from . import ml_models
from . import backtesting
from . import outlier_detection
from . import visualization
from . import utils

__all__ = [
    'data_acquisition',
    'feature_engineering',
    'hmm_regime',
    'ml_models',
    'backtesting',
    'outlier_detection',
    'visualization',
    'utils'
]
