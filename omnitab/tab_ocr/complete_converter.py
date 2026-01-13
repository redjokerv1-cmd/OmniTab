"""
Complete TAB to GP5 Converter

Combines:
1. Smart OCR for accurate note positions (string, fret)
2. Gemini Vision for rhythm analysis (duration, techniques)

This produces a complete GP5 file with both note positions AND rhythm.
"""

import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import guitarpro as gp

from .recognizer.smart_ocr import SmartTabOCR
from .recognizer.horizontal_projection import HorizontalProjection
from .recognizer.header_detector import HeaderDetector
from .recognizer.measure_detector import MeasureDetector

# Try to import Gemini
try:
    from .gemini_analyzer import GeminiTabAnalyzer, DURATION_TO_GP5
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    DURATION_TO_GP5 = {
        "whole": -2, "half": -1, "quarter": 0, 
        "eighth": 1, "sixteenth": 2, "32nd": 3
    }


@dataclass
class CompleteNote:
    """A note with both position and rhythm"""
    string: int        # 1-6
    fret: int          # 0-24
    duration: int      # GP5 format: -2 to 3
    technique: Optional[str] = None  # hammer-on, slide, bend, etc.
    x: float = 0       # X position in image
    confidence: float = 1.0


@dataclass
class CompleteBeat:
    """A beat containing one or more notes"""
    notes: List[CompleteNote]
    duration: int = 0  # GP5 duration value
    

@dataclass
class CompleteMeasure:
    """A measure containing beats"""
    number: int
    beats: List[CompleteBeat]


class CompleteConverter:
    """
    Complete TAB to GP5 converter using OCR + Gemini
    
    Pipeline:
    1. OCR: Detect note positions (string, fret, x, y)
    2. Gemini: Analyze rhythm and techniques
    3. Merge: Combine OCR positions with Gemini rhythms
    4. Generate: Create complete GP5 file
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize converter
        
        Args:
            gemini_api_key: Google AI API key for rhythm analysis.
                           If None, will use GOOGLE_API_KEY env var.
                           If not available, will use default quarter notes.
        """
        self.ocr = SmartTabOCR()
        self.line_detector = HorizontalProjection()
        self.header_detector = HeaderDetector()
        self.measure_detector = MeasureDetector()
        
        # Initialize Gemini if available
        self.gemini = None
        if GEMINI_AVAILABLE:
            try:
                self.gemini = GeminiTabAnalyzer(api_key=gemini_api_key)
                print("[Gemini] Initialized successfully")
            except Exception as e:
                print(f"[Gemini] Not available: {e}")
                print("[Gemini] Will use default quarter notes for rhythm")
    
    def convert(self,
                image_path: str,
                output_path: str,
                title: str = "OmniTab Conversion",
                use_gemini: bool = True) -> Dict:
        """
        Convert TAB image to GP5 with full rhythm support
        
        Args:
            image_path: Path to TAB image
            output_path: Path for output GP5 file
            title: Song title
            use_gemini: Whether to use Gemini for rhythm (if available)
            
        Returns:
            Conversion result with statistics
        """
        print(f"Processing: {image_path}")
        print("=" * 60)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Step 1: OCR - Get note positions
        print("\n[1] Running Smart OCR...")
        ocr_result = self.ocr.process(image)
        ocr_notes = ocr_result.get('digits', [])
        print(f"    Found {len(ocr_notes)} notes")
        
        # Step 2: Detect header (tuning, capo)
        print("\n[2] Detecting header...")
        header = self.header_detector.detect(image)
        tuning = getattr(header, 'tuning', None) or ['E', 'B', 'G', 'D', 'A', 'E']
        capo = getattr(header, 'capo', 0) or 0
        print(f"    Tuning: {tuning}, Capo: {capo}")
        
        # Step 3: Detect measures
        print("\n[3] Detecting measures...")
        systems = self.line_detector.detect(image)
        measures = self.measure_detector.detect(image, systems)
        total_measures = sum(len(m) for m in measures)
        print(f"    Found {total_measures} measures")
        
        # Step 4: Gemini rhythm analysis (if available)
        gemini_result = None
        if use_gemini and self.gemini:
            print("\n[4] Analyzing rhythm with Gemini...")
            try:
                gemini_result = self.gemini.analyze(image_path)
                if "error" not in gemini_result:
                    print(f"    Gemini analysis successful")
                else:
                    print(f"    Gemini error: {gemini_result.get('error')}")
                    gemini_result = None
            except Exception as e:
                print(f"    Gemini failed: {e}")
        else:
            print("\n[4] Skipping Gemini (not available or disabled)")
        
        # Step 5: Merge OCR + Gemini
        print("\n[5] Merging OCR positions with rhythm...")
        complete_measures = self._merge_ocr_gemini(
            ocr_notes, gemini_result, measures, systems
        )
        print(f"    Created {len(complete_measures)} measures")
        
        # Step 6: Generate GP5
        print("\n[6] Generating GP5...")
        song = self._create_gp5(complete_measures, title, tuning, capo)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gp.write(song, str(output_path))
        print(f"    Saved: {output_path}")
        
        # Count notes
        total_notes = sum(
            len(beat.notes) 
            for m in complete_measures 
            for beat in m.beats
        )
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"  Measures: {len(complete_measures)}")
        print(f"  Notes: {total_notes}")
        print(f"  Rhythm source: {'Gemini' if gemini_result else 'Default (quarter)'}")
        
        return {
            "output": str(output_path),
            "measures": len(complete_measures),
            "notes": total_notes,
            "rhythm_source": "gemini" if gemini_result else "default",
            "tuning": tuning,
            "capo": capo
        }
    
    def _merge_ocr_gemini(self,
                          ocr_notes: List,
                          gemini_result: Optional[Dict],
                          measures: List,
                          systems: List) -> List[CompleteMeasure]:
        """
        Merge OCR note positions with Gemini rhythm analysis
        
        Strategy:
        - If Gemini available: Match OCR notes to Gemini beats by position
        - If not: Group OCR notes into chords, use quarter notes
        """
        complete_measures = []
        
        if gemini_result and "measures" in gemini_result:
            # Use Gemini rhythm with OCR positions
            complete_measures = self._merge_with_gemini(ocr_notes, gemini_result)
        else:
            # Fallback: Group OCR notes into chords with default rhythm
            complete_measures = self._fallback_grouping(ocr_notes, measures, systems)
        
        return complete_measures
    
    def _merge_with_gemini(self,
                           ocr_notes: List,
                           gemini_result: Dict) -> List[CompleteMeasure]:
        """Match OCR notes to Gemini beats based on position and content"""
        complete_measures = []
        
        for gem_measure in gemini_result.get("measures", []):
            beats = []
            
            for gem_beat in gem_measure.get("beats", []):
                duration_str = gem_beat.get("duration", "quarter")
                duration_gp5 = DURATION_TO_GP5.get(duration_str, 0)
                
                notes = []
                for gem_note in gem_beat.get("notes", []):
                    string = gem_note.get("string")
                    fret = gem_note.get("fret")
                    technique = gem_note.get("technique")
                    
                    # Find matching OCR note for confidence
                    ocr_match = self._find_ocr_match(ocr_notes, string, fret)
                    confidence = ocr_match.confidence if ocr_match else 0.8
                    
                    notes.append(CompleteNote(
                        string=string,
                        fret=fret,
                        duration=duration_gp5,
                        technique=technique,
                        confidence=confidence
                    ))
                
                if notes:
                    beats.append(CompleteBeat(notes=notes, duration=duration_gp5))
            
            if beats:
                complete_measures.append(CompleteMeasure(
                    number=gem_measure.get("number", len(complete_measures) + 1),
                    beats=beats
                ))
        
        return complete_measures
    
    def _find_ocr_match(self, ocr_notes: List, string: int, fret: int):
        """Find matching OCR note by string and fret"""
        for note in ocr_notes:
            if note.string == string and note.value == fret:
                return note
        return None
    
    def _fallback_grouping(self,
                           ocr_notes: List,
                           measures: List,
                           systems: List) -> List[CompleteMeasure]:
        """Group OCR notes into measures and chords using default quarter notes"""
        complete_measures = []
        
        # Sort notes by X position
        sorted_notes = sorted(ocr_notes, key=lambda n: n.x)
        
        # Group into chords (same X position = same chord)
        chords = []
        current_chord = []
        x_threshold = 15
        
        for note in sorted_notes:
            if not current_chord:
                current_chord = [note]
            elif abs(note.x - current_chord[0].x) < x_threshold:
                current_chord.append(note)
            else:
                if current_chord:
                    chords.append(current_chord)
                current_chord = [note]
        
        if current_chord:
            chords.append(current_chord)
        
        # Create measures (4 beats per measure for 4/4 time)
        beats_per_measure = 4
        current_measure_beats = []
        measure_number = 1
        
        for chord in chords:
            # Deduplicate same string in chord
            seen_strings = set()
            unique_notes = []
            for note in sorted(chord, key=lambda n: n.string):
                if note.string not in seen_strings:
                    seen_strings.add(note.string)
                    unique_notes.append(CompleteNote(
                        string=note.string,
                        fret=note.value,
                        duration=0,  # Quarter note
                        x=note.x,
                        confidence=note.confidence
                    ))
            
            if unique_notes and len(unique_notes) <= 6:
                current_measure_beats.append(CompleteBeat(
                    notes=unique_notes,
                    duration=0
                ))
            
            # Check if measure is full
            if len(current_measure_beats) >= beats_per_measure:
                complete_measures.append(CompleteMeasure(
                    number=measure_number,
                    beats=current_measure_beats
                ))
                current_measure_beats = []
                measure_number += 1
        
        # Add remaining beats
        if current_measure_beats:
            complete_measures.append(CompleteMeasure(
                number=measure_number,
                beats=current_measure_beats
            ))
        
        return complete_measures
    
    def _create_gp5(self,
                    complete_measures: List[CompleteMeasure],
                    title: str,
                    tuning: List[str],
                    capo: int) -> gp.Song:
        """Create GP5 song from complete measures"""
        song = gp.Song()
        song.title = title
        song.tempo = 120
        
        # Use existing track from song
        track = song.tracks[0]
        track.name = "Guitar"
        track.channel.instrument = 25
        
        # Set tuning
        tuning_midi = self._tuning_to_midi(tuning)
        track.strings = [gp.GuitarString(i + 1, midi) for i, midi in enumerate(tuning_midi)]
        
        # Add capo in comments
        if capo > 0:
            try:
                song.comments = [f"Capo: Fret {capo}"]
            except:
                pass
        
        # Add measures
        for measure_idx, cm in enumerate(complete_measures):
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
            
            # Calculate duration based on beat count
            num_beats = len(cm.beats)
            if num_beats == 0:
                continue
            
            if num_beats <= 4:
                duration_value = 4  # Quarter notes
            elif num_beats <= 8:
                duration_value = 8  # Eighth notes
            else:
                duration_value = 16  # Sixteenth notes
            
            current_start = gp.Duration.quarterTime
            
            for cb in cm.beats:
                beat = gp.Beat(voice)
                beat.start = current_start
                beat.duration = gp.Duration(value=duration_value)
                beat.status = gp.BeatStatus.normal
                
                for cn in cb.notes:
                    note = gp.Note(beat)
                    note.type = gp.NoteType.normal
                    note.string = cn.string
                    note.value = cn.fret
                    note.velocity = 95
                    
                    # Apply technique
                    if cn.technique:
                        self._apply_technique(note, cn.technique)
                    
                    beat.notes.append(note)
                
                voice.beats.append(beat)
                current_start += beat.duration.time
        
        return song
    
    def _tuning_to_midi(self, tuning: List[str]) -> List[int]:
        """Convert tuning notes to MIDI values"""
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
            if note == '?' or not note:
                result.append(standard_midi[i])
            else:
                note_upper = note.upper().replace('♯', '#').replace('♭', 'B')
                base = note_to_midi.get(note_upper, 0)
                
                # Find closest octave to standard
                target = standard_midi[i]
                best_midi = target
                best_diff = float('inf')
                
                for octave in range(1, 6):
                    midi = base + (octave + 1) * 12
                    diff = abs(midi - target)
                    if diff < best_diff:
                        best_diff = diff
                        best_midi = midi
                
                result.append(best_midi)
        
        return result
    
    def _apply_technique(self, note: gp.Note, technique: str):
        """Apply technique effect to note"""
        technique = technique.lower().replace("-", "").replace("_", "")
        
        if technique in ["hammeron", "hammer"]:
            note.effect.hammer = True
        elif technique in ["pulloff", "pull"]:
            note.effect.hammer = True  # GP5 uses same flag
        elif technique in ["slide", "slideup", "slidedown"]:
            note.effect.slides = [gp.SlideType.shiftSlide]
        elif technique in ["bend"]:
            bend = gp.BendEffect()
            bend.value = 100  # Full step bend
            note.effect.bend = bend
        elif technique in ["vibrato"]:
            note.effect.vibrato = True
        elif technique in ["harmonic", "naturalharmonic"]:
            note.effect.harmonic = gp.NaturalHarmonic()
        elif technique in ["letring", "ring"]:
            note.effect.letRing = True


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python complete_converter.py <input_image> <output.gp5> [--no-gemini]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    use_gemini = "--no-gemini" not in sys.argv
    
    converter = CompleteConverter()
    result = converter.convert(
        image_path=image_path,
        output_path=output_path,
        use_gemini=use_gemini
    )
    
    print(f"\nSuccess! Created {result['output']}")
    print(f"  {result['measures']} measures, {result['notes']} notes")
    print(f"  Rhythm source: {result['rhythm_source']}")
