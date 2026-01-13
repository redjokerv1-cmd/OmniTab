"""
OCR to GP5 Converter - Convert TAB OCR results to Guitar Pro 5 format

Pipeline:
1. Run EnhancedTabOCR on image
2. Convert recognized chords to GP5 format
3. Apply tuning and capo
4. Generate GP5 file
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import guitarpro as gp

from .recognizer.enhanced_ocr import EnhancedTabOCR, TabChord, TabSystem


# Note name to MIDI pitch mapping (octave 0)
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'DB': 1,
    'D': 2, 'D#': 3, 'EB': 3,
    'E': 4, 'F': 5, 'F#': 6, 'GB': 6,
    'G': 7, 'G#': 8, 'AB': 8,
    'A': 9, 'A#': 10, 'BB': 10,
    'B': 11
}

# Standard guitar tuning MIDI pitches (strings 1-6, high to low)
STANDARD_TUNING_MIDI = [64, 59, 55, 50, 45, 40]  # E4, B3, G3, D3, A2, E2


def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI pitch."""
    note = note.upper().replace('♯', '#').replace('♭', 'B')
    base = NOTE_TO_MIDI.get(note, 0)
    return base + (octave * 12)


def tuning_to_midi(tuning: List[str]) -> List[int]:
    """
    Convert tuning note names to MIDI pitches.
    
    Args:
        tuning: List of 6 note names, e.g., ['E', 'C', 'G', 'D', 'G', 'C']
        
    Returns:
        List of 6 MIDI pitches
    """
    # Yellow Jacket tuning: ①=E ②=C ③=G ④=D ⑤=G ⑥=C
    # Standard tuning reference: E4(64), B3(59), G3(55), D3(50), A2(45), E2(40)
    # 
    # For alternate tunings, we need to find the right octave.
    # Generally, each string should be close to its standard pitch.
    
    midi_pitches = []
    
    for i, note in enumerate(tuning):
        if note == '?':
            # Unknown - use standard tuning for this string
            midi_pitches.append(STANDARD_TUNING_MIDI[i])
        else:
            # Find the closest octave to standard tuning
            standard_midi = STANDARD_TUNING_MIDI[i]
            note_base = NOTE_TO_MIDI.get(note.upper().replace('♯', '#').replace('♭', 'B'), 0)
            
            # Try different octaves and pick the closest to standard
            best_midi = None
            best_diff = float('inf')
            
            for octave in range(1, 6):
                midi = note_base + (octave + 1) * 12  # +1 because MIDI octave 0 starts at 12
                diff = abs(midi - standard_midi)
                if diff < best_diff:
                    best_diff = diff
                    best_midi = midi
            
            midi_pitches.append(best_midi if best_midi else standard_midi)
    
    return midi_pitches


class OcrToGp5Converter:
    """
    Convert TAB OCR results to Guitar Pro 5 file.
    
    Handles:
    - Fret numbers to note positions
    - Custom tuning
    - Capo position
    - Chord grouping
    """
    
    def __init__(self, use_gpu: bool = False):
        self.ocr = EnhancedTabOCR(use_gpu=use_gpu)
    
    def convert(self, 
                image_path: str,
                output_path: str,
                title: str = "Untitled",
                tempo: int = 120,
                manual_tuning: Optional[List[str]] = None,
                manual_capo: Optional[int] = None) -> Dict:
        """
        Convert TAB image to GP5 file.
        
        Args:
            image_path: Path to TAB image
            output_path: Path to output GP5 file
            title: Song title
            tempo: Tempo in BPM
            manual_tuning: Override detected tuning (6 note names)
            manual_capo: Override detected capo position
            
        Returns:
            Conversion result with stats
        """
        # Step 1: Run OCR
        print(f"Processing: {image_path}")
        result = self.ocr.process_file(image_path)
        
        # Step 2: Get tuning and capo
        tuning = manual_tuning
        capo = manual_capo
        
        if 'header' in result:
            header = result['header']
            if tuning is None:
                tuning = header.get('tuning')
            if capo is None:
                capo = header.get('capo', 0)
        
        # Default to standard tuning if not detected
        if tuning is None:
            tuning = ['E', 'B', 'G', 'D', 'A', 'E']
        if capo is None:
            capo = 0
        
        # Convert tuning to MIDI pitches
        tuning_midi = tuning_to_midi(tuning)
        
        print(f"Tuning: {tuning}")
        print(f"Tuning MIDI: {tuning_midi}")
        print(f"Capo: {capo}")
        
        # Step 3: Create GP5 song
        song = self._create_song(title, tempo)
        track = self._create_track(song, tuning_midi, capo)
        
        # Step 4: Convert chords to beats
        systems = result['systems']
        total_beats = self._add_chords_to_track(track, systems, capo)
        
        # Step 5: Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gp.write(song, str(output_path), version=(5, 1, 0))
        
        print(f"\nSaved: {output_path}")
        
        return {
            'output_path': str(output_path),
            'total_beats': total_beats,
            'total_systems': len(systems),
            'tuning': tuning,
            'capo': capo,
            'stats': result['stats']
        }
    
    def _create_song(self, title: str, tempo: int) -> gp.Song:
        """Create GP5 song."""
        song = gp.Song()
        song.title = title
        song.tempo = tempo
        return song
    
    def _create_track(self, 
                      song: gp.Song, 
                      tuning_midi: List[int],
                      capo: int) -> gp.Track:
        """Create and configure guitar track."""
        track = song.tracks[0]
        track.name = "Guitar"
        track.channel.instrument = 25  # Acoustic Guitar (steel)
        
        # Set custom tuning
        track.strings = [
            gp.GuitarString(i + 1, midi) 
            for i, midi in enumerate(tuning_midi)
        ]
        
        # Note: GP5 capo is per-track but not directly settable
        # We'll adjust fret numbers instead
        
        return track
    
    def _add_chords_to_track(self,
                              track: gp.Track,
                              systems: List[TabSystem],
                              capo: int) -> int:
        """Add all chords from all systems to the track."""
        # Get the first measure's first voice
        measure = track.measures[0]
        voice = measure.voices[0]
        
        current_start = gp.Duration.quarterTime
        beat_count = 0
        
        for system in systems:
            for chord in system.chords:
                beat = self._create_beat_from_chord(voice, chord, current_start, capo)
                voice.beats.append(beat)
                current_start += beat.duration.time
                beat_count += 1
        
        return beat_count
    
    def _create_beat_from_chord(self,
                                 voice: gp.Voice,
                                 chord: TabChord,
                                 start: int,
                                 capo: int) -> gp.Beat:
        """Create a GP5 beat from an OCR chord."""
        beat = gp.Beat(voice)
        beat.start = start
        beat.duration = gp.Duration(value=4)  # Quarter note default
        beat.status = gp.BeatStatus.normal
        
        # Add notes
        used_strings = set()
        
        for note in sorted(chord.notes, key=lambda n: n.y):
            # Determine string number based on Y position within system
            string_num = self._estimate_string_from_y(note, chord)
            
            if string_num in used_strings:
                continue
            
            used_strings.add(string_num)
            
            # Create GP5 note
            gp_note = gp.Note(beat)
            gp_note.type = gp.NoteType.normal
            gp_note.string = string_num
            
            # Adjust fret for capo
            # Note: In TAB, fret numbers are relative to capo
            # So fret 0 with capo 2 = actually fret 0 (open string above capo)
            gp_note.value = note.value
            
            beat.notes.append(gp_note)
        
        return beat
    
    def _estimate_string_from_y(self, note, chord: TabChord) -> int:
        """
        Estimate which string (1-6) based on note's Y position.
        
        Notes in a chord are vertically distributed across 6 lines.
        """
        # If chord has multiple notes, use relative Y positions
        if len(chord.notes) == 1:
            return 3  # Default to middle string for single notes
        
        # Sort notes by Y
        sorted_notes = sorted(chord.notes, key=lambda n: n.y)
        
        # Find this note's index
        idx = sorted_notes.index(note)
        
        # Map to string 1-6 (distribute evenly)
        # String 1 is highest pitch (lowest on TAB visually at top)
        # String 6 is lowest pitch (highest on TAB visually at bottom)
        total = len(sorted_notes)
        
        if total == 1:
            return 3
        elif total == 2:
            return [2, 5][idx]
        elif total == 3:
            return [1, 3, 5][idx]
        elif total == 4:
            return [1, 2, 4, 5][idx]
        elif total == 5:
            return [1, 2, 3, 4, 5][idx]
        else:
            # 6 or more - just use 1-6
            return min(idx + 1, 6)


def convert_tab_to_gp5(image_path: str,
                       output_path: str,
                       title: str = "Untitled",
                       tempo: int = 120,
                       tuning: Optional[List[str]] = None,
                       capo: Optional[int] = None,
                       use_gpu: bool = False) -> Dict:
    """
    Convenience function to convert TAB image to GP5.
    
    Args:
        image_path: Path to TAB image (PNG/JPG)
        output_path: Path to output GP5 file
        title: Song title
        tempo: Tempo in BPM
        tuning: Custom tuning (6 note names), or None for auto-detect
        capo: Capo position, or None for auto-detect
        use_gpu: Use GPU for OCR
        
    Returns:
        Conversion result dict
    """
    converter = OcrToGp5Converter(use_gpu=use_gpu)
    return converter.convert(
        image_path=image_path,
        output_path=output_path,
        title=title,
        tempo=tempo,
        manual_tuning=tuning,
        manual_capo=capo
    )


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_to_gp5.py <image_path> [output.gp5] [title] [tempo]")
        print()
        print("Example:")
        print("  python ocr_to_gp5.py tab.png output.gp5 'Yellow Jacket' 65")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.replace('.png', '.gp5')
    title = sys.argv[3] if len(sys.argv) > 3 else "Untitled"
    tempo = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    
    print("=" * 60)
    print("TAB to GP5 Converter")
    print("=" * 60)
    
    result = convert_tab_to_gp5(
        image_path=image_path,
        output_path=output_path,
        title=title,
        tempo=tempo
    )
    
    print()
    print("=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Output: {result['output_path']}")
    print(f"Beats: {result['total_beats']}")
    print(f"Systems: {result['total_systems']}")
    print(f"Tuning: {result['tuning']}")
    print(f"Capo: {result['capo']}")
