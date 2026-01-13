"""TabNote - Single note on guitar TAB"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Technique(Enum):
    """Guitar playing techniques"""
    NONE = "none"
    HAMMER_ON = "h"
    PULL_OFF = "p"
    SLIDE_UP = "/"
    SLIDE_DOWN = "\\"
    BEND = "b"
    RELEASE = "r"
    VIBRATO = "~"
    TAP = "t"
    PALM_MUTE = "pm"
    LET_RING = "let_ring"


class HarmonicType(Enum):
    """Harmonic types"""
    NONE = "none"
    NATURAL = "natural"        # <5>, <7>, <12>
    ARTIFICIAL = "artificial"  # AH
    PINCH = "pinch"           # PH
    TAP = "tap"               # TH


@dataclass
class TabNote:
    """
    Represents a single note on guitar TAB.
    
    Attributes:
        string: Guitar string number (1-6, 1 is highest/thinnest)
        fret: Fret number (0-24, 0 is open string)
        x_position: Original x coordinate in image (for timing)
        y_position: Original y coordinate in image
    """
    string: int
    fret: int
    x_position: float = 0.0
    y_position: float = 0.0
    
    # Technique
    technique: Technique = Technique.NONE
    
    # Harmonic
    harmonic_type: HarmonicType = HarmonicType.NONE
    harmonic_fret: Optional[int] = None  # For natural harmonics: <12> → 12
    
    # States
    is_muted: bool = False      # x or X or (X)
    is_ghost: bool = False      # (note) - ghost note
    is_dead: bool = False       # x - completely muted
    
    # Connection to next/previous note
    connected_to_next: bool = False  # For slides, hammer-ons, pull-offs
    
    # Confidence score (0.0 - 1.0)
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate note data"""
        if not 1 <= self.string <= 6:
            raise ValueError(f"String must be 1-6, got {self.string}")
        if not 0 <= self.fret <= 24:
            raise ValueError(f"Fret must be 0-24, got {self.fret}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for GP5Writer compatibility"""
        return {
            'string': self.string,
            'fret': self.fret,
            'technique': self.technique.value if self.technique != Technique.NONE else None,
            'harmonic': self.harmonic_type.value if self.harmonic_type != HarmonicType.NONE else None,
            'is_muted': self.is_muted,
            'is_ghost': self.is_ghost,
        }
    
    def __repr__(self) -> str:
        tech = f" ({self.technique.value})" if self.technique != Technique.NONE else ""
        harm = f" <{self.harmonic_fret}>" if self.harmonic_type == HarmonicType.NATURAL else ""
        mute = " (muted)" if self.is_muted else ""
        return f"TabNote(S{self.string}F{self.fret}{tech}{harm}{mute})"
