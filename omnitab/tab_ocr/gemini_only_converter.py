"""
Gemini-Only TAB to GP5 Converter

Skip OCR entirely - let Gemini Vision analyze everything:
- Note positions (string, fret)
- Rhythm/duration
- Techniques
- Tuning, capo, tempo

This is simpler and potentially more accurate than OCR + Gemini hybrid.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import guitarpro as gp

logger = logging.getLogger(__name__)

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    from .gemini_analyzer import GeminiTabAnalyzer, DURATION_TO_GP5
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    DURATION_TO_GP5 = {}


class GeminiOnlyConverter:
    """
    Convert TAB images to GP5 using Gemini Vision only.
    
    No OCR - Gemini does everything:
    1. Read the TAB image
    2. Identify all notes (string, fret)
    3. Determine rhythm/duration
    4. Detect techniques
    5. Extract tuning, capo, tempo
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini not available. Install: pip install google-generativeai")
        
        self.analyzer = GeminiTabAnalyzer(api_key=api_key)
        logger.info("[GeminiOnly] Initialized")
    
    def convert(
        self,
        image_path: str,
        output_path: str,
        title: str = "OmniTab (Gemini)",
        tempo: int = 120
    ) -> Dict:
        """
        Convert TAB image to GP5 using Gemini only
        
        Args:
            image_path: Path to TAB image (PNG, JPG)
            output_path: Output GP5 file path
            title: Song title
            tempo: Default tempo (used if Gemini doesn't detect)
            
        Returns:
            Dict with conversion results
        """
        logger.info(f"[GeminiOnly] Converting: {image_path}")
        print("=" * 60)
        print(f"[GeminiOnly] Analyzing with Gemini Vision...")
        print("=" * 60)
        
        # Step 1: Analyze with Gemini
        print("\n[1] Sending to Gemini Vision API...")
        try:
            result = self.analyzer.analyze(image_path)
            print("    [OK] Received response")
        except Exception as e:
            logger.error(f"[GeminiOnly] Gemini failed: {e}")
            raise ValueError(f"Gemini analysis failed: {e}")
        
        # Extract info
        measures_data = result.get("measures", [])
        tuning = result.get("tuning", ["E", "B", "G", "D", "A", "E"])
        capo = result.get("capo", 0)
        detected_tempo = result.get("tempo", tempo)
        
        print(f"\n[2] Gemini Results:")
        print(f"    Measures: {len(measures_data)}")
        print(f"    Tuning: {tuning}")
        print(f"    Capo: {capo}")
        print(f"    Tempo: {detected_tempo}")
        
        # Count notes
        total_notes = sum(
            len(beat.get("notes", []))
            for m in measures_data
            for beat in m.get("beats", [])
        )
        print(f"    Total Notes: {total_notes}")
        
        # Step 2: Create GP5
        print("\n[3] Generating GP5...")
        song = self._create_gp5(measures_data, title, detected_tempo, tuning, capo)
        
        # Step 3: Save
        gp.write(song, output_path)
        print(f"    [OK] Saved: {output_path}")
        
        # Count beats in GP5
        gp5_beats = sum(
            len(v.beats) 
            for t in song.tracks 
            for m in t.measures 
            for v in m.voices
        )
        
        print("\n" + "=" * 60)
        print("RESULT (Gemini Only)")
        print("=" * 60)
        print(f"  Measures: {len(measures_data)}")
        print(f"  Beats: {gp5_beats}")
        print(f"  Notes: {total_notes}")
        print(f"  Tuning: {tuning}")
        print(f"  Capo: {capo}")
        print(f"  Tempo: {detected_tempo}")
        
        return {
            "output": output_path,
            "measures": len(measures_data),
            "notes": total_notes,
            "beats": gp5_beats,
            "rhythm_source": "gemini",
            "tuning": tuning,
            "capo": capo,
            "tempo": detected_tempo,
            "mode": "gemini_only"
        }
    
    def _create_gp5(
        self,
        measures_data: List[Dict],
        title: str,
        tempo: int,
        tuning: List[str],
        capo: int
    ) -> gp.Song:
        """Create GP5 song from Gemini analysis"""
        song = gp.Song()
        song.title = title
        song.artist = "OmniTab (Gemini Only)"
        song.tempo = tempo
        
        # Add capo info to instructions (since track.offset doesn't persist)
        if capo > 0:
            song.instructions = f"Capo: {capo} fret"
        
        # Create track
        track = gp.Track(song)
        track.name = "Acoustic Guitar"
        track.channel = self._create_channel()
        track.strings = self._create_strings(tuning)
        track.fretCount = 24
        track.offset = capo  # Capo setting! This is the correct way
        
        # Process measures
        for i, m_data in enumerate(measures_data):
            # Create measure header
            header = gp.MeasureHeader()
            header.number = i + 1
            header.start = 960 * i * 4
            header.timeSignature = gp.TimeSignature(4, gp.Duration(1))
            song.measureHeaders.append(header)
            
            # Create measure
            measure = gp.Measure(track, header)
            
            # Process beats
            beats = m_data.get("beats", [])
            current_start = 0
            
            for b_data in beats:
                duration_str = b_data.get("duration", "quarter")
                duration_val = DURATION_TO_GP5.get(duration_str, 0)
                
                beat = gp.Beat(measure.voices[0])
                beat.start = current_start
                beat.duration = gp.Duration(value=2**abs(duration_val) if duration_val <= 0 else 2**duration_val)
                
                # Add notes
                notes = b_data.get("notes", [])
                used_strings = set()
                
                for n_data in notes:
                    string = n_data.get("string")
                    fret = n_data.get("fret")
                    technique = n_data.get("technique")
                    
                    # Validate
                    if string is None or fret is None:
                        continue
                    
                    # Handle 'X' and other non-numeric
                    if isinstance(fret, str):
                        if fret.upper() == 'X':
                            continue  # Skip muted
                        try:
                            fret = int(fret)
                        except ValueError:
                            continue
                    
                    if not (1 <= string <= 6) or not (0 <= fret <= 24):
                        continue
                    
                    if string in used_strings:
                        continue
                    used_strings.add(string)
                    
                    note = gp.Note(beat)
                    note.string = string
                    note.value = fret
                    note.velocity = 95
                    note.type = gp.NoteType.normal
                    
                    # Apply technique
                    if technique:
                        self._apply_technique(note, technique)
                    
                    beat.notes.append(note)
                
                if beat.notes:
                    beat.status = gp.BeatStatus.normal
                else:
                    beat.status = gp.BeatStatus.rest
                
                measure.voices[0].beats.append(beat)
                current_start += 960  # Quarter note default
            
            # Ensure at least one beat
            if not measure.voices[0].beats:
                rest = gp.Beat(measure.voices[0])
                rest.status = gp.BeatStatus.rest
                rest.duration = gp.Duration(value=1)
                measure.voices[0].beats.append(rest)
            
            track.measures.append(measure)
        
        # Handle empty measures case
        if not track.measures:
            header = gp.MeasureHeader()
            header.number = 1
            song.measureHeaders.append(header)
            
            measure = gp.Measure(track, header)
            rest = gp.Beat(measure.voices[0])
            rest.status = gp.BeatStatus.rest
            rest.duration = gp.Duration(value=1)
            measure.voices[0].beats.append(rest)
            track.measures.append(measure)
        
        song.tracks.append(track)
        return song
    
    def _create_channel(self) -> gp.MidiChannel:
        """Create MIDI channel for guitar"""
        channel = gp.MidiChannel()
        channel.channel = 0
        channel.effectChannel = 1
        channel.instrument = 25  # Steel guitar
        channel.volume = 100
        channel.balance = 64
        channel.chorus = 0
        channel.reverb = 0
        channel.phaser = 0
        channel.tremolo = 0
        return channel
    
    def _create_strings(self, tuning: List[str]) -> List[gp.GuitarString]:
        """Create guitar strings with tuning"""
        note_to_midi = {
            'C': 0, 'C#': 1, 'DB': 1,
            'D': 2, 'D#': 3, 'EB': 3,
            'E': 4, 'F': 5, 'F#': 6, 'GB': 6,
            'G': 7, 'G#': 8, 'AB': 8,
            'A': 9, 'A#': 10, 'BB': 10,
            'B': 11
        }
        
        # Standard tuning MIDI values
        standard = [64, 59, 55, 50, 45, 40]
        
        strings = []
        for i, note in enumerate(tuning[:6]):
            if note == '?' or not note:
                midi = standard[i]
            else:
                note_upper = note.upper().replace('♯', '#').replace('♭', 'B')
                base = note_to_midi.get(note_upper, 0)
                
                # Find closest octave
                target = standard[i]
                best = target
                best_diff = float('inf')
                
                for octave in range(1, 6):
                    midi = base + (octave + 1) * 12
                    diff = abs(midi - target)
                    if diff < best_diff:
                        best_diff = diff
                        best = midi
                
                midi = best
            
            strings.append(gp.GuitarString(i + 1, midi))
        
        return strings
    
    def _apply_technique(self, note: gp.Note, technique: str):
        """Apply technique effect to note (comprehensive implementation)"""
        if not technique:
            return
        
        tech_lower = technique.lower()
        
        # Hammer-on / Pull-off (H/P)
        if 'hammer' in tech_lower or tech_lower == 'h':
            note.effect.hammer = True
        elif 'pull' in tech_lower or tech_lower == 'p':
            note.effect.hammer = True  # GP5 uses same flag for H/P
        
        # Slide
        elif 'slide' in tech_lower or tech_lower in ['/', '\\', 's']:
            note.effect.slides = [gp.SlideType.shiftSlide]
        
        # Bend
        elif 'bend' in tech_lower or tech_lower == 'b':
            note.effect.bend = gp.BendEffect(
                type=gp.BendType.bend,
                value=100,
                points=[
                    gp.BendPoint(0, 0),
                    gp.BendPoint(6, 100),
                    gp.BendPoint(12, 100)
                ]
            )
        
        # Vibrato
        elif 'vibrato' in tech_lower or tech_lower == 'v':
            note.effect.vibrato = True
        
        # Natural Harmonic
        elif 'harmonic' in tech_lower or 'harm' in tech_lower:
            note.effect.harmonic = gp.NaturalHarmonic()
        
        # Let Ring
        elif 'let ring' in tech_lower or 'ring' in tech_lower:
            note.effect.letRing = True
        
        # Palm Mute
        elif 'palm' in tech_lower or 'mute' in tech_lower or tech_lower == 'pm':
            note.effect.palmMute = True
        
        # Staccato
        elif 'staccato' in tech_lower:
            note.effect.staccato = True


# Quick test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m omnitab.tab_ocr.gemini_only_converter <image> <output.gp5>")
        sys.exit(1)
    
    converter = GeminiOnlyConverter()
    result = converter.convert(sys.argv[1], sys.argv[2])
    print(f"\nDone! Notes: {result['notes']}")
