"""TabChord - Multiple notes played simultaneously"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .tab_note import TabNote


class StrumDirection(Enum):
    """Strum direction for chords"""
    NONE = "none"
    DOWN = "down"
    UP = "up"


class Duration(Enum):
    """Note/chord duration (in Guitar Pro notation)"""
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHTH = 8
    SIXTEENTH = 16
    THIRTY_SECOND = 32
    SIXTY_FOURTH = 64


@dataclass
class TabChord:
    """
    Represents notes played at the same time (chord or single note).
    
    In TAB, notes at the same x-position are played together.
    """
    notes: List[TabNote] = field(default_factory=list)
    x_position: float = 0.0
    
    # Timing
    duration: Duration = Duration.QUARTER
    is_dotted: bool = False
    is_triplet: bool = False
    
    # Rest
    is_rest: bool = False
    
    # Strum
    strum_direction: StrumDirection = StrumDirection.NONE
    
    # Confidence score
    confidence: float = 1.0
    
    @property
    def is_single_note(self) -> bool:
        """Check if this is a single note (not a chord)"""
        return len(self.notes) == 1
    
    @property
    def is_chord(self) -> bool:
        """Check if this has multiple notes"""
        return len(self.notes) > 1
    
    def add_note(self, note: TabNote) -> None:
        """Add a note to this chord"""
        self.notes.append(note)
        # Update position to average
        if len(self.notes) > 0:
            self.x_position = sum(n.x_position for n in self.notes) / len(self.notes)
    
    def get_notes_sorted_by_string(self) -> List[TabNote]:
        """Get notes sorted by string number (1-6)"""
        return sorted(self.notes, key=lambda n: n.string)
    
    def to_notes_data(self) -> dict:
        """
        Convert to format compatible with GP5Writer.
        
        Returns dict with 'pitches' or 'frets' for each note.
        """
        if self.is_rest:
            return {
                'type': 'rest',
                'duration': self.duration.value
            }
        
        return {
            'type': 'chord' if self.is_chord else 'note',
            'duration': self.duration.value,
            'notes': [n.to_dict() for n in self.notes]
        }
    
    def __repr__(self) -> str:
        if self.is_rest:
            return f"TabChord(rest, {self.duration.name})"
        notes_str = ", ".join(f"S{n.string}F{n.fret}" for n in self.notes)
        return f"TabChord([{notes_str}], {self.duration.name})"
