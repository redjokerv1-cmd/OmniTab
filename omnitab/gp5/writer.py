"""Guitar Pro 5 file writer using PyGuitarPro."""

from pathlib import Path

import guitarpro as gp


class GP5Writer:
    """
    Writes note data to Guitar Pro 5 format.

    Uses PyGuitarPro library for file generation.
    """

    # Standard guitar tuning (E A D G B E)
    STANDARD_TUNING = [
        gp.GuitarString(1, 64),  # E4
        gp.GuitarString(2, 59),  # B3
        gp.GuitarString(3, 55),  # G3
        gp.GuitarString(4, 50),  # D3
        gp.GuitarString(5, 45),  # A2
        gp.GuitarString(6, 40),  # E2
    ]

    def __init__(self, title: str = "Untitled", tempo: int = 120):
        self.title = title
        self.tempo = tempo

    def write(self, notes_data: list[dict], output_path: Path) -> None:
        """
        Write notes data to GP5 file.

        Args:
            notes_data: Normalized notes data
            output_path: Path to output GP5 file
        """
        song = self._create_song()
        track = self._create_track(song)

        # Use existing measure and voice (Song creates default)
        measure = track.measures[0]
        voice = measure.voices[0]

        # Add all notes to the first voice
        for note_data in notes_data:
            self._add_note_to_voice(voice, note_data)

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

    def _add_note_to_voice(self, voice: gp.Voice, note_data: dict) -> None:
        """Add a note or chord to a voice."""
        beat = gp.Beat(voice)

        if note_data["type"] == "rest":
            beat.status = gp.BeatStatus.rest
        elif note_data["type"] == "note":
            note = self._create_note(beat, note_data)
            beat.notes.append(note)
        elif note_data["type"] == "chord":
            for pitch in note_data["pitches"]:
                note = self._create_note(beat, {"pitch": pitch, "effects": []})
                beat.notes.append(note)

        voice.beats.append(beat)

    def _create_note(self, beat: gp.Beat, note_data: dict) -> gp.Note:
        """Create a GP5 note with effects."""
        note = gp.Note(beat)

        # Calculate fret and string from MIDI pitch
        pitch = note_data["pitch"]
        string_num, fret = self._midi_to_guitar_position(pitch)
        note.string = string_num
        note.value = fret

        # Apply effects
        for effect in note_data.get("effects", []):
            self._apply_effect(note, effect)

        return note

    def _midi_to_guitar_position(self, midi_pitch: int) -> tuple[int, int]:
        """
        Convert MIDI pitch to guitar string and fret.

        Simple implementation - picks first available position.
        TODO: Implement smarter fingering algorithm.
        """
        # String open pitches (high to low): E4(64), B3(59), G3(55), D3(50), A2(45), E2(40)
        string_pitches = [64, 59, 55, 50, 45, 40]

        for string_num, open_pitch in enumerate(string_pitches, 1):
            fret = midi_pitch - open_pitch
            if 0 <= fret <= 22:  # Valid fret range
                return string_num, fret

        # Default to 6th string if no suitable position found
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

    def _apply_bend(self, note: gp.Note, effect: dict) -> None:
        """Apply bend effect."""
        bend = gp.BendEffect()
        bend.type = gp.BendType.bend
        # Default to whole step bend
        bend.points = [
            gp.BendPoint(0, 0),
            gp.BendPoint(60, 100),  # 100 = whole step
        ]
        note.effect.bend = bend

    def _apply_slide(self, note: gp.Note, effect: dict) -> None:
        """Apply slide effect."""
        if effect["type"] == "slide_up":
            note.effect.slides.append(gp.SlideType.shiftSlideTo)
        else:
            note.effect.slides.append(gp.SlideType.shiftSlideTo)
