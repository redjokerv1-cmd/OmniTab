"""
OCR to GP5 v2 - Using Horizontal Projection for accurate line mapping

This version uses the correct approach:
1. Detect TAB lines with horizontal projection
2. Map OCR digits to exact string numbers
3. Group into chords and measures
4. Generate valid GP5 file
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import guitarpro as gp

from .recognizer.enhanced_ocr import EnhancedTabOCR
from .recognizer.horizontal_projection import HorizontalProjection, TabStaffSystem
from .recognizer.header_detector import HeaderDetector
from .recognizer.measure_detector import MeasureDetector, Measure


@dataclass
class MappedNote:
    """A note with accurate string mapping"""
    fret: int
    string: int  # 1-6
    x: float
    y: float
    confidence: float


@dataclass
class MappedChord:
    """A chord with properly mapped notes"""
    notes: List[MappedNote]
    x_position: float
    
    @property
    def is_valid(self) -> bool:
        """Valid if max 6 notes and no duplicate strings"""
        if len(self.notes) > 6:
            return False
        strings = [n.string for n in self.notes]
        return len(strings) == len(set(strings))


class OcrToGp5V2:
    """
    Convert TAB image to GP5 using horizontal projection.
    
    Key improvement: Accurate string number mapping
    """
    
    def __init__(self, use_gpu: bool = False):
        self.ocr = EnhancedTabOCR(use_gpu=use_gpu)
        self.line_detector = HorizontalProjection()
        self.header_detector = HeaderDetector(use_gpu=use_gpu)
        self.measure_detector = MeasureDetector()
    
    def convert(self,
                image_path: str,
                output_path: str,
                title: str = "Untitled",
                tempo: int = 120,
                manual_tuning: Optional[List[str]] = None,
                manual_capo: Optional[int] = None) -> Dict:
        """Convert TAB image to GP5"""
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read: {image_path}")
        
        print(f"Processing: {image_path}")
        print("=" * 60)
        
        # Step 1: Detect TAB lines (horizontal projection)
        print("\n[1] Detecting TAB lines...")
        systems = self.line_detector.detect(image)
        print(f"    Found {len(systems)} TAB systems")
        
        if not systems:
            raise ValueError("No TAB systems detected!")
        
        # Step 2: Run OCR
        print("\n[2] Running OCR...")
        ocr_result = self.ocr.process(image)
        digits = ocr_result['digits']
        print(f"    Found {len(digits)} digits")
        
        # Step 3: Detect header (tuning, capo)
        print("\n[3] Detecting header...")
        try:
            header = self.header_detector.detect(image)
            detected_tuning = header.tuning
            detected_capo = header.capo
        except:
            detected_tuning = ['E', 'B', 'G', 'D', 'A', 'E']
            detected_capo = 0
        
        tuning = manual_tuning or detected_tuning
        capo = manual_capo if manual_capo is not None else detected_capo
        print(f"    Tuning: {tuning}")
        print(f"    Capo: {capo}")
        
        # Step 4: Detect measures
        print("\n[4] Detecting measures...")
        all_measures = self.measure_detector.detect(image, systems)
        total_measures = sum(len(m) for m in all_measures)
        print(f"    Found: {total_measures} measures across {len(systems)} systems")
        
        # Step 5: Map digits to strings
        print("\n[5] Mapping digits to strings...")
        mapped_notes = self._map_digits_to_strings(digits, systems)
        print(f"    Mapped: {len(mapped_notes)} notes")
        
        # Step 6: Group into chords
        print("\n[6] Grouping into chords...")
        chords = self._group_into_chords(mapped_notes)
        valid_chords = [c for c in chords if c.is_valid]
        print(f"    Total chords: {len(chords)}")
        print(f"    Valid chords: {len(valid_chords)}")
        
        # Step 7: Assign chords to measures
        print("\n[7] Assigning chords to measures...")
        chords_per_measure = self._assign_chords_to_measures(valid_chords, systems, all_measures)
        
        # Step 8: Create GP5
        print("\n[8] Creating GP5 file...")
        song = self._create_gp5_with_measures(chords_per_measure, title, tempo, tuning)
        
        # Step 7: Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gp.write(song, str(output_path), version=(5, 1, 0))
        print(f"    Saved: {output_path}")
        
        return {
            'output_path': str(output_path),
            'systems_detected': len(systems),
            'digits_recognized': len(digits),
            'notes_mapped': len(mapped_notes),
            'chords_total': len(chords),
            'chords_valid': len(valid_chords),
            'tuning': tuning,
            'capo': capo
        }
    
    def _map_digits_to_strings(self,
                                digits,
                                systems: List[TabStaffSystem]) -> List[MappedNote]:
        """Map OCR digits to string numbers using line positions"""
        mapped = []
        
        for digit in digits:
            # Find which system
            for system in systems:
                if system.y_start <= digit.y <= system.y_end:
                    string_num = system.get_string_for_y(digit.y)
                    if string_num > 0:
                        mapped.append(MappedNote(
                            fret=digit.value,
                            string=string_num,
                            x=digit.x,
                            y=digit.y,
                            confidence=digit.confidence
                        ))
                    break
        
        return mapped
    
    def _group_into_chords(self,
                           notes: List[MappedNote],
                           x_threshold: float = 15) -> List[MappedChord]:
        """Group notes into chords by X position"""
        if not notes:
            return []
        
        sorted_notes = sorted(notes, key=lambda n: n.x)
        chords = []
        current = [sorted_notes[0]]
        
        for note in sorted_notes[1:]:
            if note.x - current[-1].x < x_threshold:
                current.append(note)
            else:
                chords.append(MappedChord(
                    notes=current,
                    x_position=sum(n.x for n in current) / len(current)
                ))
                current = [note]
        
        if current:
            chords.append(MappedChord(
                notes=current,
                x_position=sum(n.x for n in current) / len(current)
            ))
        
        return chords
    
    def _assign_chords_to_measures(self,
                                    chords: List[MappedChord],
                                    systems: List[TabStaffSystem],
                                    all_measures: List[List[Measure]]) -> List[List[MappedChord]]:
        """Assign chords to their respective measures based on X position"""
        chords_per_measure = []
        
        for sys_idx, system in enumerate(systems):
            if sys_idx >= len(all_measures):
                continue
            
            measures = all_measures[sys_idx]
            
            # Get chords in this system
            system_chords = [c for c in chords 
                           if any(system.y_start <= n.y <= system.y_end for n in c.notes)]
            
            for measure in measures:
                # Get chords in this measure
                measure_chords = [c for c in system_chords
                                 if measure.x_start <= c.x_position <= measure.x_end]
                
                # Sort by X position
                measure_chords.sort(key=lambda c: c.x_position)
                
                if measure_chords:
                    chords_per_measure.append(measure_chords)
        
        return chords_per_measure
    
    def _create_gp5_with_measures(self,
                                   chords_per_measure: List[List[MappedChord]],
                                   title: str,
                                   tempo: int,
                                   tuning: List[str]) -> gp.Song:
        """Create GP5 with proper measure structure"""
        song = gp.Song()
        song.title = title
        song.tempo = tempo
        
        track = song.tracks[0]
        track.name = "Guitar"
        track.channel.instrument = 25
        
        # Set tuning
        tuning_midi = self._tuning_to_midi(tuning)
        track.strings = [gp.GuitarString(i + 1, midi) for i, midi in enumerate(tuning_midi)]
        
        # Add measures
        for measure_idx, measure_chords in enumerate(chords_per_measure):
            # Ensure we have enough measures
            while len(track.measures) <= measure_idx:
                if measure_idx > 0:
                    header = gp.MeasureHeader()
                    song.measureHeaders.append(header)
                    for t in song.tracks:
                        new_measure = gp.Measure(t, header)
                        t.measures.append(new_measure)
            
            measure = track.measures[measure_idx]
            voice = measure.voices[0]
            
            # Distribute chords evenly across the measure
            num_chords = len(measure_chords)
            if num_chords == 0:
                continue
            
            # Calculate beat duration based on number of chords
            # Assume 4/4 time
            if num_chords <= 4:
                duration_value = 4  # Quarter notes
            elif num_chords <= 8:
                duration_value = 8  # Eighth notes
            else:
                duration_value = 16  # Sixteenth notes
            
            current_start = gp.Duration.quarterTime
            
            for chord in measure_chords:
                beat = gp.Beat(voice)
                beat.start = current_start
                beat.duration = gp.Duration(value=duration_value)
                beat.status = gp.BeatStatus.normal
                
                for note_data in sorted(chord.notes, key=lambda n: n.string):
                    note = gp.Note(beat)
                    note.type = gp.NoteType.normal
                    note.string = note_data.string
                    note.value = note_data.fret
                    beat.notes.append(note)
                
                voice.beats.append(beat)
                current_start += beat.duration.time
        
        return song
    
    def _create_gp5(self,
                    chords: List[MappedChord],
                    title: str,
                    tempo: int,
                    tuning: List[str]) -> gp.Song:
        """Create GP5 song from chords"""
        song = gp.Song()
        song.title = title
        song.tempo = tempo
        
        track = song.tracks[0]
        track.name = "Guitar"
        track.channel.instrument = 25  # Acoustic Guitar
        
        # Set tuning
        tuning_midi = self._tuning_to_midi(tuning)
        track.strings = [
            gp.GuitarString(i + 1, midi)
            for i, midi in enumerate(tuning_midi)
        ]
        
        # Create beats in the first measure
        # GP5 structure: Song has measures, each measure has voices, each voice has beats
        measure = track.measures[0]
        voice = measure.voices[0]
        
        current_start = gp.Duration.quarterTime
        
        for chord in chords:
            beat = gp.Beat(voice)
            beat.start = current_start
            beat.duration = gp.Duration(value=4)  # Quarter note
            beat.status = gp.BeatStatus.normal
            
            # Add notes (sorted by string)
            for note_data in sorted(chord.notes, key=lambda n: n.string):
                note = gp.Note(beat)
                note.type = gp.NoteType.normal
                note.string = note_data.string
                note.value = note_data.fret
                beat.notes.append(note)
            
            voice.beats.append(beat)
            current_start += beat.duration.time
        
        return song
    
    def _tuning_to_midi(self, tuning: List[str]) -> List[int]:
        """Convert tuning to MIDI pitches"""
        note_to_midi = {
            'C': 0, 'C#': 1, 'DB': 1,
            'D': 2, 'D#': 3, 'EB': 3,
            'E': 4, 'F': 5, 'F#': 6, 'GB': 6,
            'G': 7, 'G#': 8, 'AB': 8,
            'A': 9, 'A#': 10, 'BB': 10,
            'B': 11
        }
        
        standard_midi = [64, 59, 55, 50, 45, 40]
        result = []
        
        for i, note in enumerate(tuning):
            if note == '?':
                result.append(standard_midi[i])
            else:
                note_upper = note.upper().replace('♯', '#').replace('♭', 'B')
                base = note_to_midi.get(note_upper, 0)
                
                # Find octave closest to standard
                best_midi = standard_midi[i]
                best_diff = float('inf')
                
                for octave in range(1, 6):
                    midi = base + (octave + 1) * 12
                    diff = abs(midi - standard_midi[i])
                    if diff < best_diff:
                        best_diff = diff
                        best_midi = midi
                
                result.append(best_midi)
        
        return result


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_to_gp5_v2.py <image> [output.gp5] [title] [tempo]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else image_path.replace('.png', '_v2.gp5')
    title = sys.argv[3] if len(sys.argv) > 3 else "Untitled"
    tempo = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    
    converter = OcrToGp5V2()
    result = converter.convert(image_path, output_path, title, tempo)
    
    print()
    print("=" * 60)
    print("CONVERSION RESULT")
    print("=" * 60)
    print(f"Output: {result['output_path']}")
    print(f"Systems: {result['systems_detected']}")
    print(f"Digits: {result['digits_recognized']}")
    print(f"Notes mapped: {result['notes_mapped']}")
    print(f"Valid chords: {result['chords_valid']}/{result['chords_total']}")
