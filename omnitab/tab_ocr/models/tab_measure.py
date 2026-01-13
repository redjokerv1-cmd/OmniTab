"""TabMeasure - A single measure/bar in TAB"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .tab_chord import TabChord


@dataclass
class TabMeasure:
    """
    Represents a single measure (bar) in guitar TAB.
    """
    number: int = 1
    chords: List[TabChord] = field(default_factory=list)
    
    # Time signature
    time_signature: Tuple[int, int] = (4, 4)  # (beats, beat_value)
    
    # Position in image
    x_start: float = 0.0
    x_end: float = 0.0
    
    # Repeat signs
    has_repeat_start: bool = False
    has_repeat_end: bool = False
    repeat_count: int = 1
    
    @property
    def beat_count(self) -> int:
        """Number of beats in this measure"""
        return self.time_signature[0]
    
    @property
    def beat_value(self) -> int:
        """Beat value (4 = quarter, 8 = eighth)"""
        return self.time_signature[1]
    
    @property
    def chord_count(self) -> int:
        """Number of chords/notes in this measure"""
        return len(self.chords)
    
    def add_chord(self, chord: TabChord) -> None:
        """Add a chord to this measure"""
        self.chords.append(chord)
    
    def get_chords_sorted(self) -> List[TabChord]:
        """Get chords sorted by x position (time order)"""
        return sorted(self.chords, key=lambda c: c.x_position)
    
    def to_notes_data(self) -> List[dict]:
        """Convert to list of note data for GP5Writer"""
        return [chord.to_notes_data() for chord in self.get_chords_sorted()]
    
    def __repr__(self) -> str:
        ts = f"{self.time_signature[0]}/{self.time_signature[1]}"
        return f"TabMeasure({self.number}, {ts}, {self.chord_count} chords)"
