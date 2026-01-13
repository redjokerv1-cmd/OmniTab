"""Guitar Pro 5 file writer using PyGuitarPro."""

from pathlib import Path
from typing import Optional

import guitarpro as gp


class GP5Writer:
    """
    Writes note data to Guitar Pro 5 format.

    Uses PyGuitarPro library for file generation.
    """

    # Standard guitar tuning (E A D G B E) - high to low
    STANDARD_TUNING = [
        gp.GuitarString(1, 64),  # E4
        gp.GuitarString(2, 59),  # B3
        gp.GuitarString(3, 55),  # G3
        gp.GuitarString(4, 50),  # D3
        gp.GuitarString(5, 45),  # A2
        gp.GuitarString(6, 40),  # E2
    ]

    # String open pitches for fret calculation
    STRING_PITCHES = [64, 59, 55, 50, 45, 40]

    def __init__(self, title: str = "Untitled", tempo: int = 120):
        self.title = title
        self.tempo = tempo

    def write(self, notes_data: list[dict], output_path: Path) -> None:
        """
        Write notes data to GP5 file.

        Args:
            notes_data: Normalized notes data, each with:
                - type: 'note', 'chord', or 'rest'
                - pitch: MIDI pitch (for 'note')
                - pitches: list of MIDI pitches (for 'chord')
                - duration: beat duration value (1=whole, 2=half, 4=quarter, etc.)
                - effects: list of effect dicts
            output_path: Path to output GP5 file
        """
        song = self._create_song()
        track = self._create_track(song)

        # Use existing measure and voice (Song creates default)
        measure = track.measures[0]
        voice = measure.voices[0]

        # Track current beat position
        current_start = gp.Duration.quarterTime  # Start at 960 (quarter note time)

        # Add all notes/chords/rests to the voice
        for note_data in notes_data:
            beat = self._create_beat(voice, note_data, current_start)
            voice.beats.append(beat)
            # Advance position by beat duration
            current_start += beat.duration.time

        # Write to file (GP5 format)
        gp.write(song, str(output_path), version=(5, 1, 0))

    def _create_song(self) -> gp.Song:
        """Create a new GP5 song."""
        song = gp.Song()
        song.title = self.title
        song.tempo = self.tempo
        return song

    def _create_track(self, song: gp.Song) -> gp.Track:
        """Configure the default guitar track with standard tuning."""
        # Song already has a default track, use it
        track = song.tracks[0]
        track.name = "Guitar"
        track.strings = self.STANDARD_TUNING.copy()
        track.channel.instrument = 25  # Acoustic Guitar (steel)
        return track

    def _create_beat(self, voice: gp.Voice, note_data: dict, start: int) -> gp.Beat:
        """Create a beat with notes or rest."""
        beat = gp.Beat(voice)
        beat.start = start

        # Set duration (default to quarter note)
        duration_value = note_data.get("duration", 4)
        beat.duration = gp.Duration(value=duration_value)

        note_type = note_data.get("type", "note")

        if note_type == "rest":
            beat.status = gp.BeatStatus.rest
        elif note_type == "note":
            beat.status = gp.BeatStatus.normal
            note = self._create_note(beat, note_data)
            if note:
                beat.notes.append(note)
        elif note_type == "chord":
            beat.status = gp.BeatStatus.normal
            # Track used strings to avoid duplicates
            used_strings = set()
            for pitch in note_data.get("pitches", []):
                note = self._create_note(
                    beat, 
                    {"pitch": pitch, "effects": note_data.get("effects", [])},
                    exclude_strings=used_strings
                )
                if note:
                    used_strings.add(note.string)
                    beat.notes.append(note)

        return beat

    def _create_note(
        self, 
        beat: gp.Beat, 
        note_data: dict,
        exclude_strings: Optional[set] = None
    ) -> Optional[gp.Note]:
        """Create a GP5 note with effects."""
        pitch = note_data.get("pitch")
        if pitch is None:
            return None

        note = gp.Note(beat)
        note.type = gp.NoteType.normal

        # Calculate fret and string from MIDI pitch
        string_num, fret = self._midi_to_guitar_position(pitch, exclude_strings)
        note.string = string_num
        note.value = fret

        # Apply effects
        for effect in note_data.get("effects", []):
            self._apply_effect(note, effect)

        return note

    def _midi_to_guitar_position(
        self, 
        midi_pitch: int,
        exclude_strings: Optional[set] = None
    ) -> tuple[int, int]:
        """
        Convert MIDI pitch to guitar string and fret.

        Args:
            midi_pitch: MIDI note number
            exclude_strings: Set of string numbers to avoid (for chords)

        Returns:
            Tuple of (string_number, fret)
        """
        exclude_strings = exclude_strings or set()

        # Try each string from high to low (prefer lower fret positions)
        for string_num, open_pitch in enumerate(self.STRING_PITCHES, 1):
            if string_num in exclude_strings:
                continue
            fret = midi_pitch - open_pitch
            if 0 <= fret <= 22:  # Valid fret range
                return string_num, fret

        # If no valid position found on available strings, find the best available string
        # even if it means using a high fret or skipping the note
        best_string = None
        best_fret = None
        min_fret_diff = float('inf')
        
        for string_num, open_pitch in enumerate(self.STRING_PITCHES, 1):
            if string_num in exclude_strings:
                continue
            fret = midi_pitch - open_pitch
            # Find the closest valid fret position
            if fret < 0:
                # Note is below this string's range
                continue
            elif fret > 22:
                # Note is above this string's range, but it's still a candidate
                # if it's the closest to valid range
                fret_diff = fret - 22
                if fret_diff < min_fret_diff:
                    min_fret_diff = fret_diff
                    best_string = string_num
                    best_fret = 22  # Clamp to max fret
            else:
                # Valid fret found
                return string_num, fret
        
        # If we found a string that's close, use it with clamped fret
        if best_string is not None:
            return best_string, best_fret

        # Last resort: find any available string
        available_strings = set(range(1, 7)) - exclude_strings
        if available_strings:
            # Pick the lowest available string
            string_num = max(available_strings)
            fret = max(0, min(22, midi_pitch - self.STRING_PITCHES[string_num - 1]))
            return string_num, fret

        # Absolute fallback - use string 6 regardless
        return 6, max(0, min(22, midi_pitch - 40))

    def _apply_effect(self, note: gp.Note, effect: dict) -> None:
        """Apply effect to GP5 note."""
        effect_type = effect.get("type", "")

        if effect_type == "ghost_note":
            note.effect.ghostNote = True
        elif effect_type == "palm_mute":
            note.effect.palmMute = True
        elif effect_type == "dead_note":
            note.type = gp.NoteType.dead
        elif effect_type == "hammer_on" or effect_type == "pull_off":
            note.effect.hammer = True
        elif effect_type == "vibrato":
            note.effect.vibrato = True
        elif effect_type == "staccato":
            note.effect.staccato = True
        elif effect_type == "let_ring":
            note.effect.letRing = True
        elif effect_type == "bend":
            self._apply_bend(note, effect)
        elif effect_type in ("slide_up", "slide_down"):
            self._apply_slide(note, effect)
        elif effect_type == "harmonic":
            self._apply_harmonic(note, effect)
        elif effect_type == "accent":
            note.effect.accentuatedNote = True
        elif effect_type == "heavy_accent":
            note.effect.heavyAccentuatedNote = True

    def _apply_bend(self, note: gp.Note, effect: dict) -> None:
        """Apply bend effect."""
        bend = gp.BendEffect()
        bend.type = gp.BendType.bend
        # Default to whole step bend (100 = 1 semitone)
        bend_value = effect.get("value", 200)  # 200 = whole step (2 semitones)
        bend.points = [
            gp.BendPoint(0, 0),
            gp.BendPoint(gp.BendEffect.maxPosition, bend_value),
        ]
        note.effect.bend = bend

    def _apply_slide(self, note: gp.Note, effect: dict) -> None:
        """Apply slide effect."""
        if effect.get("type") == "slide_up":
            note.effect.slides.append(gp.SlideType.shiftSlideTo)
        else:
            note.effect.slides.append(gp.SlideType.shiftSlideTo)

    def _apply_harmonic(self, note: gp.Note, effect: dict) -> None:
        """Apply harmonic effect."""
        harmonic_type = effect.get("harmonic_type", "natural")
        if harmonic_type == "natural":
            note.effect.harmonic = gp.NaturalHarmonic()
        elif harmonic_type == "artificial":
            note.effect.harmonic = gp.ArtificialHarmonic()
        elif harmonic_type == "pinch":
            note.effect.harmonic = gp.PinchHarmonic()
        elif harmonic_type == "tapped":
            fret = effect.get("fret", note.value + 12)
            note.effect.harmonic = gp.TappedHarmonic(fret=fret)
