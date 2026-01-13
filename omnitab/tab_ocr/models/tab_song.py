"""TabSong - Complete guitar TAB song"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .tab_measure import TabMeasure


# Standard guitar tuning (string 1-6, high to low)
STANDARD_TUNING = ['E', 'B', 'G', 'D', 'A', 'E']

# Standard tuning MIDI pitches (string 1-6)
STANDARD_TUNING_MIDI = [64, 59, 55, 50, 45, 40]


@dataclass
class TabSong:
    """
    Represents a complete guitar TAB song.
    """
    # Metadata
    title: str = "Untitled"
    artist: str = ""
    album: str = ""
    transcriber: str = "OmniTab"
    
    # Guitar settings
    tuning: List[str] = field(default_factory=lambda: STANDARD_TUNING.copy())
    tuning_midi: List[int] = field(default_factory=lambda: STANDARD_TUNING_MIDI.copy())
    capo: int = 0
    
    # Tempo
    tempo: int = 120
    
    # Content
    measures: List[TabMeasure] = field(default_factory=list)
    
    # Source info
    source_file: Optional[str] = None
    page_count: int = 0
    
    @property
    def measure_count(self) -> int:
        """Total number of measures"""
        return len(self.measures)
    
    @property
    def total_chords(self) -> int:
        """Total number of chords across all measures"""
        return sum(m.chord_count for m in self.measures)
    
    def add_measure(self, measure: TabMeasure) -> None:
        """Add a measure to the song"""
        measure.number = len(self.measures) + 1
        self.measures.append(measure)
    
    def set_tuning(self, tuning_str: List[str]) -> None:
        """
        Set tuning from string notation.
        
        Args:
            tuning_str: List of note names, e.g. ['E', 'C', 'G', 'D', 'G', 'C']
        """
        self.tuning = tuning_str
        
        # Convert to MIDI pitches
        note_to_midi = {
            'C': 48, 'C#': 49, 'Db': 49,
            'D': 50, 'D#': 51, 'Eb': 51,
            'E': 52, 'F': 53, 'F#': 54, 'Gb': 54,
            'G': 55, 'G#': 56, 'Ab': 56,
            'A': 57, 'A#': 58, 'Bb': 58,
            'B': 59
        }
        
        # Calculate MIDI pitches for each string
        # String 1 is typically high E (E4 = 64)
        base_octaves = [4, 3, 3, 3, 2, 2]  # Typical octaves for 6-string guitar
        
        self.tuning_midi = []
        for i, note in enumerate(tuning_str):
            base_midi = note_to_midi.get(note, 48)
            octave_offset = base_octaves[i] * 12
            self.tuning_midi.append(base_midi + octave_offset - 48)
    
    def to_notes_data(self) -> List[dict]:
        """
        Convert entire song to notes_data format for GP5Writer.
        
        Returns list of note events compatible with GP5Writer.write()
        """
        notes_data = []
        
        for measure in self.measures:
            for chord in measure.get_chords_sorted():
                if chord.is_rest:
                    notes_data.append({
                        'type': 'rest',
                        'duration': chord.duration.value
                    })
                else:
                    for note in chord.notes:
                        # Convert fret/string to MIDI pitch
                        string_idx = note.string - 1
                        if string_idx < len(self.tuning_midi):
                            midi_pitch = self.tuning_midi[string_idx] + note.fret
                        else:
                            midi_pitch = 60 + note.fret  # Default to middle C
                        
                        note_data = {
                            'pitch': midi_pitch,
                            'duration': chord.duration.value,
                            'string': note.string,
                            'fret': note.fret,
                        }
                        
                        # Add technique effects
                        if note.technique.value != 'none':
                            note_data['technique'] = note.technique.value
                        
                        if note.harmonic_type.value != 'none':
                            note_data['harmonic'] = note.harmonic_type.value
                            if note.harmonic_fret:
                                note_data['harmonic_fret'] = note.harmonic_fret
                        
                        if note.is_muted:
                            note_data['muted'] = True
                        
                        notes_data.append(note_data)
        
        return notes_data
    
    def get_stats(self) -> dict:
        """Get statistics about the song"""
        total_notes = 0
        techniques = {}
        
        for measure in self.measures:
            for chord in measure.chords:
                for note in chord.notes:
                    total_notes += 1
                    tech = note.technique.value
                    techniques[tech] = techniques.get(tech, 0) + 1
        
        return {
            'title': self.title,
            'measures': self.measure_count,
            'chords': self.total_chords,
            'notes': total_notes,
            'tempo': self.tempo,
            'capo': self.capo,
            'tuning': self.tuning,
            'techniques': techniques
        }
    
    def __repr__(self) -> str:
        return (f"TabSong('{self.title}' by {self.artist}, "
                f"{self.measure_count} measures, {self.total_chords} chords)")
