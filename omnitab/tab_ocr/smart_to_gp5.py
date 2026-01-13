"""
Smart OCR to GP5 - Using sliding window OCR for accurate conversion

This uses the SmartTabOCR which properly handles 2-digit fret numbers.
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import cv2
import numpy as np
import guitarpro as gp

from .recognizer.smart_ocr import SmartTabOCR, SmartDigit
from .recognizer.horizontal_projection import TabStaffSystem
from .recognizer.header_detector import HeaderDetector
from .recognizer.measure_detector import MeasureDetector, Measure

# Learning DB
try:
    from ..learning.db import LearningDB, get_image_hash
    from ..learning.models import OCRAttempt
    LEARNING_ENABLED = True
except ImportError:
    LEARNING_ENABLED = False


@dataclass
class SmartChord:
    """Chord from Smart OCR"""
    notes: List[SmartDigit]
    x_position: float
    
    @property
    def is_valid(self) -> bool:
        if len(self.notes) > 6:
            return False
        strings = [n.string for n in self.notes]
        return len(strings) == len(set(strings))


class SmartToGp5:
    """
    Convert TAB to GP5 using Smart OCR.
    
    Smart OCR uses sliding windows to properly detect 2-digit frets.
    """
    
    def __init__(self, use_gpu: bool = False):
        self.ocr = SmartTabOCR(min_confidence=0.3)
        self.header_detector = HeaderDetector(use_gpu=use_gpu)
        self.measure_detector = MeasureDetector()
        
        # Learning DB
        self.learning_db = None
        if LEARNING_ENABLED:
            try:
                self.learning_db = LearningDB()
            except Exception:
                pass
    
    def convert(self,
                image_path: str,
                output_path: str,
                title: str = "OmniTab",
                tempo: int = 120,
                manual_tuning: Optional[List[str]] = None,
                manual_capo: Optional[int] = None) -> Dict:
        """Convert image to GP5"""
        
        print(f"Processing: {image_path}")
        print("=" * 60)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load {image_path}")
        
        # Step 1: Smart OCR
        print("\n[1] Running Smart OCR (sliding window)...")
        ocr_result = self.ocr.process(image)
        digits = ocr_result['digits']
        systems = ocr_result['systems']
        
        print(f"    Systems: {len(systems)}")
        print(f"    Raw detections: {ocr_result['stats']['raw_detections']}")
        print(f"    After merge: {len(digits)}")
        print(f"    2-digit frets: {sum(1 for d in digits if d.value >= 10)}")
        
        # Step 2: Header detection
        print("\n[2] Detecting header...")
        try:
            header = self.header_detector.detect(image)
            detected_tuning = header.tuning
            detected_capo = header.capo
        except Exception:
            detected_tuning = ['E', 'B', 'G', 'D', 'A', 'E']
            detected_capo = 0
        
        tuning = manual_tuning or detected_tuning
        capo = manual_capo if manual_capo is not None else detected_capo
        print(f"    Tuning: {tuning}")
        print(f"    Capo: {capo}")
        
        # Step 3: Detect measures
        print("\n[3] Detecting measures...")
        all_measures = self.measure_detector.detect(image, systems)
        total_measures = sum(len(m) for m in all_measures)
        print(f"    Found: {total_measures} measures")
        
        # Step 4: Group into chords
        print("\n[4] Grouping into chords...")
        chords = self._group_into_chords(digits)
        valid_chords = [c for c in chords if c.is_valid]
        print(f"    Total: {len(chords)}, Valid: {len(valid_chords)}")
        
        # Step 5: Assign to measures
        print("\n[5] Assigning to measures...")
        chords_per_measure = self._assign_to_measures(valid_chords, systems, all_measures)
        
        # Step 6: Create GP5
        print("\n[6] Creating GP5...")
        song = self._create_gp5(chords_per_measure, title, tempo, tuning, capo)
        
        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gp.write(song, str(output_path), version=(5, 1, 0))
        print(f"    Saved: {output_path}")
        
        # Count notes in GP5
        gp5_notes = sum(
            len(beat.notes)
            for measure in song.tracks[0].measures
            for beat in measure.voices[0].beats
            if beat.notes
        )
        
        result = {
            'output_path': str(output_path),
            'systems': len(systems),
            'raw_detections': ocr_result['stats']['raw_detections'],
            'merged_digits': len(digits),
            'two_digit_frets': sum(1 for d in digits if d.value >= 10),
            'chords_total': len(chords),
            'chords_valid': len(valid_chords),
            'measures': total_measures,
            'gp5_notes': gp5_notes,
            'tuning': tuning,
            'capo': capo
        }
        
        # Save to DB
        if self.learning_db:
            try:
                attempt = OCRAttempt(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    image_path=str(image_path),
                    image_hash=get_image_hash(str(image_path)),
                    total_digits=ocr_result['stats']['raw_detections'],
                    mapped_digits=len(digits),
                    unmapped_digits=0,  # Smart OCR only returns mapped
                    duplicates_removed=ocr_result['stats']['raw_detections'] - len(digits),
                    systems_detected=len(systems),
                    measures_detected=total_measures,
                    avg_confidence=sum(d.confidence for d in digits) / max(len(digits), 1),
                    suspicious_count=sum(1 for d in digits if d.value > 20),
                    gp5_path=str(output_path),
                    gp5_notes=gp5_notes,
                    gp5_measures=len(chords_per_measure),
                    settings={'method': 'smart_ocr', 'tuning': tuning, 'capo': capo}
                )
                self.learning_db.save_attempt(attempt)
                print(f"\n    [DB] Saved attempt {attempt.id[:8]}...")
            except Exception as e:
                print(f"\n    [DB] Failed: {e}")
        
        print("\n" + "=" * 60)
        print("CONVERSION RESULT")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Systems: {len(systems)}")
        print(f"Digits: {len(digits)} (2-digit: {result['two_digit_frets']})")
        print(f"Chords: {len(valid_chords)}/{len(chords)}")
        print(f"GP5 Notes: {gp5_notes}")
        
        return result
    
    def _group_into_chords(self,
                           digits: List[SmartDigit],
                           x_threshold: float = 12) -> List[SmartChord]:
        """Group digits into chords by X position"""
        if not digits:
            return []
        
        sorted_digits = sorted(digits, key=lambda d: d.x)
        
        chords = []
        current = [sorted_digits[0]]
        
        for digit in sorted_digits[1:]:
            if digit.x - current[-1].x < x_threshold:
                current.append(digit)
            else:
                chords.append(SmartChord(
                    notes=current,
                    x_position=sum(d.x for d in current) / len(current)
                ))
                current = [digit]
        
        if current:
            chords.append(SmartChord(
                notes=current,
                x_position=sum(d.x for d in current) / len(current)
            ))
        
        return chords
    
    def _assign_to_measures(self,
                            chords: List[SmartChord],
                            systems: List[TabStaffSystem],
                            all_measures: List[List[Measure]]) -> List[List[SmartChord]]:
        """Assign chords to measures"""
        result = []
        
        for sys_idx, system in enumerate(systems):
            if sys_idx >= len(all_measures):
                continue
            
            measures = all_measures[sys_idx]
            
            # Get chords in this system
            system_chords = [c for c in chords
                           if any(system.y_start <= n.y <= system.y_end for n in c.notes)]
            
            for measure in measures:
                measure_chords = [c for c in system_chords
                                 if measure.x_start <= c.x_position <= measure.x_end]
                measure_chords.sort(key=lambda c: c.x_position)
                
                if measure_chords:
                    result.append(measure_chords)
        
        return result
    
    def _create_gp5(self,
                    chords_per_measure: List[List[SmartChord]],
                    title: str,
                    tempo: int,
                    tuning: List[str],
                    capo: int) -> gp.Song:
        """Create GP5 file"""
        song = gp.Song()
        song.title = title
        song.tempo = tempo
        
        track = song.tracks[0]
        track.name = "Guitar"
        track.channel.instrument = 25
        
        # Set tuning
        tuning_midi = self._tuning_to_midi(tuning)
        track.strings = [gp.GuitarString(i + 1, midi) for i, midi in enumerate(tuning_midi)]
        
        # Capo in comments
        if capo > 0:
            try:
                song.comments = [f"Capo: Fret {capo}"]
            except:
                pass
        
        # Add measures
        for measure_idx, measure_chords in enumerate(chords_per_measure):
            while len(track.measures) <= measure_idx:
                if measure_idx > 0:
                    header = gp.MeasureHeader()
                    song.measureHeaders.append(header)
                    for t in song.tracks:
                        new_measure = gp.Measure(t, header)
                        t.measures.append(new_measure)
            
            measure = track.measures[measure_idx]
            voice = measure.voices[0]
            
            num_chords = len(measure_chords)
            if num_chords == 0:
                continue
            
            # Duration based on chord count
            if num_chords <= 4:
                duration_value = 4
            elif num_chords <= 8:
                duration_value = 8
            else:
                duration_value = 16
            
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
                    note.value = note_data.value
                    beat.notes.append(note)
                
                voice.beats.append(beat)
                current_start += beat.duration.time
        
        return song
    
    def _tuning_to_midi(self, tuning: List[str]) -> List[int]:
        """Convert tuning to MIDI"""
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
                octave = 4 if i < 2 else 3 if i < 4 else 2
                result.append(base + (octave + 1) * 12)
        
        return result


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python smart_to_gp5.py <input_image> <output.gp5> [title] [tempo]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "OmniTab"
    tempo = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    
    converter = SmartToGp5()
    result = converter.convert(input_path, output_path, title, tempo)
