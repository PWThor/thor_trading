import os
import sys
import logging
from .feature_generator import FeatureGenerator
from . import config

logger = logging.getLogger(__name__)

def create_feature_generator():
    """Create and return a feature generator with default configuration"""
    return FeatureGenerator(config)

# Example of use
if __name__ == "__main__":
    feature_gen = create_feature_generator()
    print("Feature generator created successfully")