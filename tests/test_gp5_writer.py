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
                "duration": 1.0,
                "offset": 0.0,
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

        # Cleanup
        output_path.unlink()

    def test_create_chord(self, writer):
        # Use notes on different strings to avoid overlap
        notes_data = [
            {
                "type": "chord",
                "pitches": [40, 47, 52],  # E2, B2, E3 - different strings
                "duration": 1.0,
                "offset": 0.0,
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

        output_path.unlink()

    def test_create_rest(self, writer):
        notes_data = [
            {
                "type": "rest",
                "duration": 1.0,
                "offset": 0.0,
            }
        ]

        with tempfile.NamedTemporaryFile(suffix=".gp5", delete=False) as f:
            output_path = Path(f.name)

        writer.write(notes_data, output_path)

        song = gp.parse(str(output_path))
        assert output_path.exists()

        output_path.unlink()

    def test_apply_palm_mute(self, writer):
        notes_data = [
            {
                "type": "note",
                "pitch": 60,
                "duration": 1.0,
                "offset": 0.0,
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
                "duration": 1.0,
                "offset": 0.0,
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
