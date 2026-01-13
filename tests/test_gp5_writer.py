"""Tests for GP5Writer."""

import tempfile
from pathlib import Path

import pytest
import guitarpro as gp

from omnitab.gp5.writer import GP5Writer


class TestGP5Writer:
    """Test cases for GP5 file generation."""

    @pytest.fixture
    def writer(self):
        return GP5Writer(title="Test Song", tempo=120)

    def test_create_simple_note(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 60,  # C4
                "duration": 4,  # Quarter note
                "effects": [],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        # Verify file was created
        assert output_path.exists()

        # Read and verify content
        song = gp.parse(str(output_path))
        assert song.title == "Test Song"
        assert song.tempo == 120
        assert len(song.tracks) == 1
        
        # Verify beat structure
        voice = song.tracks[0].measures[0].voices[0]
        assert len(voice.beats) == 1
        assert len(voice.beats[0].notes) == 1
        assert voice.beats[0].status == gp.BeatStatus.normal

        # Cleanup
        output_path.unlink()

    def test_create_multiple_notes(self, writer):
        """Test that multiple notes create separate beats."""
        notes_data = [
            {"type": "note", "pitch": 64, "duration": 4, "effects": []},
            {"type": "note", "pitch": 66, "duration": 4, "effects": []},
            {"type": "note", "pitch": 68, "duration": 4, "effects": []},
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        voice = song.tracks[0].measures[0].voices[0]
        
        # Should have 3 separate beats
        assert len(voice.beats) == 3
        assert all(len(b.notes) == 1 for b in voice.beats)

        output_path.unlink()

    def test_create_chord(self, writer):
        # Use notes on different strings to avoid overlap
        notes_data = [
            {
                "type": "chord",
                "pitches": [40, 47, 52],  # E2, B2, E3 - different strings
                "duration": 4,
                "effects": [],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        track = song.tracks[0]
        measure = track.measures[0]
        beat = measure.voices[0].beats[0]

        assert len(beat.notes) == 3
        
        # Verify notes are on different strings
        strings = {note.string for note in beat.notes}
        assert len(strings) == 3  # All on different strings

        output_path.unlink()

    def test_create_rest(self, writer):
        notes_data = [
            {
                "type": "rest",
                "duration": 4,
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        voice = song.tracks[0].measures[0].voices[0]
        
        assert len(voice.beats) == 1
        assert voice.beats[0].status == gp.BeatStatus.rest

        output_path.unlink()

    def test_mixed_notes_and_rests(self, writer):
        """Test mixed notes, rests, and chords."""
        notes_data = [
            {"type": "note", "pitch": 64, "duration": 8, "effects": []},
            {"type": "rest", "duration": 8},
            {"type": "note", "pitch": 67, "duration": 8, "effects": []},
            {"type": "chord", "pitches": [40, 47, 52], "duration": 4, "effects": []},
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        voice = song.tracks[0].measures[0].voices[0]
        
        assert len(voice.beats) == 4
        assert voice.beats[0].status == gp.BeatStatus.normal
        assert voice.beats[1].status == gp.BeatStatus.rest
        assert voice.beats[2].status == gp.BeatStatus.normal
        assert voice.beats[3].status == gp.BeatStatus.normal
        assert len(voice.beats[3].notes) == 3  # Chord

        output_path.unlink()

    def test_apply_palm_mute(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 60,
                "duration": 4,
                "effects": [{"type": "palm_mute"}],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
        assert note.effect.palmMute is True

        output_path.unlink()

    def test_apply_ghost_note(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 60,
                "duration": 4,
                "effects": [{"type": "ghost_note"}],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
        assert note.effect.ghostNote is True

        output_path.unlink()

    def test_apply_hammer_on(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 64,
                "duration": 4,
                "effects": [{"type": "hammer_on"}],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
        assert note.effect.hammer is True

        output_path.unlink()

    def test_apply_vibrato(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 64,
                "duration": 4,
                "effects": [{"type": "vibrato"}],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
        assert note.effect.vibrato is True

        output_path.unlink()

    def test_apply_bend(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 64,
                "duration": 4,
                "effects": [{"type": "bend", "value": 200}],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
        assert note.effect.bend is not None
        assert len(note.effect.bend.points) == 2

        output_path.unlink()

    def test_apply_harmonic(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 64,
                "duration": 4,
                "effects": [{"type": "harmonic", "harmonic_type": "natural"}],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
        assert note.effect.harmonic is not None
        assert isinstance(note.effect.harmonic, gp.NaturalHarmonic)

        output_path.unlink()

    def test_midi_to_guitar_position(self, writer):
        # E2 (low E string open)
        string, fret = writer._midi_to_guitar_position(40)
        assert string == 6
        assert fret == 0

        # A2 (A string open)
        string, fret = writer._midi_to_guitar_position(45)
        assert string == 5
        assert fret == 0

        # E4 (high E string open)
        string, fret = writer._midi_to_guitar_position(64)
        assert string == 1
        assert fret == 0

        # C5 (high E string 8th fret)
        string, fret = writer._midi_to_guitar_position(72)
        assert string == 1
        assert fret == 8

    def test_chord_no_string_overlap(self, writer):
        """Test that chord notes don't overlap on the same string."""
        # E major chord - all 6 strings
        notes_data = [
            {
                "type": "chord",
                "pitches": [40, 47, 52, 56, 59, 64],  # E major open
                "duration": 1,
                "effects": [],
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        beat = song.tracks[0].measures[0].voices[0].beats[0]
        
        # Should have 6 notes, all on different strings
        assert len(beat.notes) == 6
        strings = [note.string for note in beat.notes]
        assert len(set(strings)) == 6  # All unique strings

        output_path.unlink()

    def test_different_durations(self, writer):
        """Test different note duration values."""
        notes_data = [
            {"type": "note", "pitch": 64, "duration": 1, "effects": []},   # Whole note
            {"type": "note", "pitch": 66, "duration": 2, "effects": []},   # Half note
            {"type": "note", "pitch": 68, "duration": 4, "effects": []},   # Quarter note
            {"type": "note", "pitch": 70, "duration": 8, "effects": []},   # Eighth note
            {"type": "note", "pitch": 72, "duration": 16, "effects": []},  # Sixteenth note
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        beats = song.tracks[0].measures[0].voices[0].beats
        
        assert len(beats) == 5
        assert beats[0].duration.value == 1
        assert beats[1].duration.value == 2
        assert beats[2].duration.value == 4
        assert beats[3].duration.value == 8
        assert beats[4].duration.value == 16

        output_path.unlink()
