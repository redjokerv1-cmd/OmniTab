"""Tests for NotationDetector."""

import pytest
from omnitab.notation.detector import NotationDetector


class TestNotationDetector:
    """Test cases for notation detection."""

    @pytest.fixture
    def detector(self):
        return NotationDetector()

    # Chord detection tests
    def test_detect_major_chord(self, detector):
        result = detector.detect("C")
        assert result.notation_type == "chord_major"
        assert result.confidence >= 0.85

    def test_detect_minor_chord(self, detector):
        result = detector.detect("Am")
        assert result.notation_type == "chord_minor"
        assert result.confidence >= 0.85

    def test_detect_major7_chord(self, detector):
        result = detector.detect("Cmaj7")
        assert result.notation_type == "chord_major7"
        assert result.confidence >= 0.85

    def test_detect_major7_triangle(self, detector):
        result = detector.detect("C△7")
        assert result.notation_type == "chord_major7"
        assert result.confidence >= 0.85

    def test_detect_dom7_chord(self, detector):
        result = detector.detect("G7")
        assert result.notation_type == "chord_dom7"
        assert result.confidence >= 0.85

    # Effect detection tests
    def test_detect_bend(self, detector):
        result = detector.detect("5b7")
        assert result.notation_type == "bend"
        assert result.confidence >= 0.85

    def test_detect_slide_up(self, detector):
        result = detector.detect("5/7")
        assert result.notation_type == "slide_up"
        assert result.confidence >= 0.85

    def test_detect_hammer_on(self, detector):
        result = detector.detect("5h7")
        assert result.notation_type == "hammer_on"
        assert result.confidence >= 0.85

    def test_detect_pull_off(self, detector):
        result = detector.detect("7p5")
        assert result.notation_type == "pull_off"
        assert result.confidence >= 0.85

    def test_detect_dead_note(self, detector):
        result = detector.detect("x")
        assert result.notation_type == "dead_note"
        assert result.confidence >= 0.85

    def test_detect_ghost_note(self, detector):
        result = detector.detect("(5)")
        assert result.notation_type == "ghost_note"
        assert result.confidence >= 0.85

    def test_detect_vibrato(self, detector):
        result = detector.detect("5~")
        assert result.notation_type == "vibrato"
        assert result.confidence >= 0.85

    def test_detect_palm_mute(self, detector):
        result = detector.detect("PM")
        assert result.notation_type == "palm_mute"
        assert result.confidence >= 0.85

    # Confidence threshold tests
    def test_auto_processable(self, detector):
        result = detector.detect("Cmaj7")
        assert detector.is_auto_processable(result)

    def test_unknown_notation(self, detector):
        result = detector.detect("???unknown???")
        assert result.notation_type == "unknown"
        assert result.confidence == 0.0
        assert detector.needs_manual_selection(result)
