"""
Learning Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import json


@dataclass
class OCRAttempt:
    """Record of a single OCR attempt"""
    id: str
    timestamp: datetime
    image_path: str
    image_hash: str  # For identifying same image
    
    # OCR Results
    total_digits: int
    mapped_digits: int
    unmapped_digits: int
    duplicates_removed: int
    
    # TAB Detection
    systems_detected: int
    measures_detected: int
    
    # Quality Metrics
    avg_confidence: float
    suspicious_count: int  # Frets > 19
    
    # Output
    gp5_path: Optional[str] = None
    gp5_notes: int = 0
    gp5_measures: int = 0
    
    # Settings used
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # User feedback (if any)
    user_rating: Optional[int] = None  # 1-5
    user_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'image_path': self.image_path,
            'image_hash': self.image_hash,
            'total_digits': self.total_digits,
            'mapped_digits': self.mapped_digits,
            'unmapped_digits': self.unmapped_digits,
            'duplicates_removed': self.duplicates_removed,
            'systems_detected': self.systems_detected,
            'measures_detected': self.measures_detected,
            'avg_confidence': self.avg_confidence,
            'suspicious_count': self.suspicious_count,
            'gp5_path': self.gp5_path,
            'gp5_notes': self.gp5_notes,
            'gp5_measures': self.gp5_measures,
            'settings': self.settings,
            'user_rating': self.user_rating,
            'user_notes': self.user_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OCRAttempt':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ErrorPattern:
    """Identified error pattern for future prevention"""
    id: str
    pattern_type: str  # 'duplicate', 'wrong_fret', 'wrong_string', 'missed'
    description: str
    
    # Context
    image_region: Optional[str] = None  # Cropped image of error
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    
    # Location
    x: float = 0
    y: float = 0
    string: int = 0
    
    # Frequency
    occurrences: int = 1
    
    # Resolution
    resolved: bool = False
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'image_region': self.image_region,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'x': self.x,
            'y': self.y,
            'string': self.string,
            'occurrences': self.occurrences,
            'resolved': self.resolved,
            'resolution': self.resolution
        }


@dataclass
class Correction:
    """User correction for learning"""
    id: str
    attempt_id: str
    timestamp: datetime
    
    # What was wrong
    original_string: int
    original_fret: int
    original_x: float
    original_y: float
    
    # What it should be
    correct_string: int
    correct_fret: int
    
    # Context
    measure_num: int = 0
    beat_num: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'attempt_id': self.attempt_id,
            'timestamp': self.timestamp.isoformat(),
            'original_string': self.original_string,
            'original_fret': self.original_fret,
            'original_x': self.original_x,
            'original_y': self.original_y,
            'correct_string': self.correct_string,
            'correct_fret': self.correct_fret,
            'measure_num': self.measure_num,
            'beat_num': self.beat_num
        }
