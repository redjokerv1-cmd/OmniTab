"""
OmniTab Learning Module

Stores all OCR attempts, results, and error patterns for continuous improvement.
"""

from .db import LearningDB
from .models import OCRAttempt, ErrorPattern, Correction

__all__ = ['LearningDB', 'OCRAttempt', 'ErrorPattern', 'Correction']
