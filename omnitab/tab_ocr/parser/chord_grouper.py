"""ChordGrouper - Group notes played simultaneously into chords"""

from typing import List
from dataclasses import dataclass

from ..models import TabNote, TabChord, Duration
from ..recognizer.position_mapper import MappedNote


class ChordGrouper:
    """
    Group notes that are played simultaneously into chords.
    
    Notes at the same x-position are considered to be played together.
    """
    
    def __init__(self, 
                 x_threshold: float = 15,
                 default_duration: Duration = Duration.QUARTER):
        """
        Initialize ChordGrouper.
        
        Args:
            x_threshold: Max x distance to consider same position
            default_duration: Default note duration
        """
        self.x_threshold = x_threshold
        self.default_duration = default_duration
    
    def group(self, mapped_notes: List[MappedNote]) -> List[TabChord]:
        """
        Group mapped notes into chords.
        
        Args:
            mapped_notes: List of notes with position info
            
        Returns:
            List of TabChord objects, sorted by x position
        """
        if not mapped_notes:
            return []
        
        # Sort by x position
        sorted_notes = sorted(mapped_notes, key=lambda m: m.note.x_position)
        
        chords = []
        current_notes = [sorted_notes[0].note]
        current_x = sorted_notes[0].note.x_position
        confidences = [sorted_notes[0].digit.confidence]
        
        for mapped in sorted_notes[1:]:
            note = mapped.note
            
            if note.x_position - current_x < self.x_threshold:
                # Same chord
                current_notes.append(note)
                confidences.append(mapped.digit.confidence)
                # Update x to average
                current_x = sum(n.x_position for n in current_notes) / len(current_notes)
            else:
                # New chord - save current
                chord = self._create_chord(current_notes, current_x, confidences)
                chords.append(chord)
                
                # Start new chord
                current_notes = [note]
                current_x = note.x_position
                confidences = [mapped.digit.confidence]
        
        # Don't forget last chord
        if current_notes:
            chord = self._create_chord(current_notes, current_x, confidences)
            chords.append(chord)
        
        return chords
    
    def group_notes(self, notes: List[TabNote]) -> List[TabChord]:
        """
        Group TabNote objects directly into chords.
        
        Args:
            notes: List of TabNote objects
            
        Returns:
            List of TabChord objects
        """
        if not notes:
            return []
        
        # Sort by x position
        sorted_notes = sorted(notes, key=lambda n: n.x_position)
        
        chords = []
        current_notes = [sorted_notes[0]]
        current_x = sorted_notes[0].x_position
        
        for note in sorted_notes[1:]:
            if note.x_position - current_x < self.x_threshold:
                current_notes.append(note)
                current_x = sum(n.x_position for n in current_notes) / len(current_notes)
            else:
                chord = TabChord(
                    notes=current_notes.copy(),
                    x_position=current_x,
                    duration=self.default_duration,
                    confidence=min(n.confidence for n in current_notes)
                )
                chords.append(chord)
                
                current_notes = [note]
                current_x = note.x_position
        
        # Last chord
        if current_notes:
            chord = TabChord(
                notes=current_notes.copy(),
                x_position=current_x,
                duration=self.default_duration,
                confidence=min(n.confidence for n in current_notes)
            )
            chords.append(chord)
        
        return chords
    
    def _create_chord(self, 
                      notes: List[TabNote],
                      x_position: float,
                      confidences: List[float]) -> TabChord:
        """Create a TabChord from notes"""
        # Sort notes by string (1 = highest)
        sorted_notes = sorted(notes, key=lambda n: n.string)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        
        return TabChord(
            notes=sorted_notes,
            x_position=x_position,
            duration=self.default_duration,
            confidence=avg_confidence
        )
    
    def validate_chord(self, chord: TabChord) -> bool:
        """
        Validate a chord for guitar playability.
        
        Checks:
        - No duplicate strings
        - All frets within reasonable span
        """
        if not chord.notes:
            return chord.is_rest
        
        # Check for duplicate strings
        strings = [n.string for n in chord.notes]
        if len(strings) != len(set(strings)):
            return False
        
        # Check fret span (max 4 frets typical, allow 5)
        frets = [n.fret for n in chord.notes if not n.is_muted and n.fret > 0]
        if frets:
            span = max(frets) - min(frets)
            if span > 5:
                return False
        
        return True
