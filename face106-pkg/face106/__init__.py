"""face106: 106-point facial landmark detection.

Quick usage:
    from face106 import LandmarkDetector
    detector = LandmarkDetector()
    landmarks = detector.predict(image, bbox)  # (106, 2) pixel coords
"""
from .detector import LandmarkDetector

__version__ = "0.1.0"
__all__ = ["LandmarkDetector"]
