"""Notation normalizer for converting various formats to GP5 standard."""

import json
from pathlib import Path
from typing import Any

from omnitab.notation.detector import NotationDetector


class NotationNormalizer:
    """
    Normalizes various notation formats to Guitar Pro 5 standard.

    Supports:
    - Chord symbols (jazz, pop, classical)
    - Effect notations (bend, slide, hammer-on, etc.)
    - Fingering notations
    """

    def __init__(self, mapping_file: Path = None):
        self.detector = NotationDetector()
        self.mapping = self._load_mapping(mapping_file)

    def _load_mapping(self, mapping_file: Path = None) -> dict:
        """Load notation mapping from JSON file."""
        if mapping_file and mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Default mapping
        return {
            "chords": {
                "major-seventh": ["maj7", "M7", "△7", "Δ7"],
                "minor-seventh": ["m7", "-7", "min7"],
                "dominant-seventh": ["7", "dom7"],
                "diminished": ["dim", "o", "°"],
                "augmented": ["+", "aug"],
                "suspended-second": ["sus2"],
                "suspended-fourth": ["sus4", "sus"],
            },
            "effects": {
                "bend": ["b", "^", "/bend"],
                "slide_up": ["/", "s", "S"],
                "slide_down": ["\\"],
                "hammer_on": ["h", "H"],
                "pull_off": ["p", "P"],
                "vibrato": ["~", "vib"],
                "palm_mute": ["PM", "P.M.", "pm"],
                "ghost_note": ["()", "ghost"],
                "dead_note": ["x", "X", "mute"],
            },
        }

    def normalize(self, notes_data: list[dict]) -> list[dict]:
        """
        Normalize all notations in notes data.

        Args:
            notes_data: List of note dictionaries from OMR

        Returns:
            Normalized notes data with GP5-compatible effects
        """
        normalized = []

        for note_data in notes_data:
            normalized_note = note_data.copy()

            # Normalize effects if present
            if "effects" in note_data:
                normalized_note["effects"] = [
                    self._normalize_effect(effect) for effect in note_data["effects"]
                ]

            # Normalize chord names if present
            if note_data.get("type") == "chord" and "chord_symbol" in note_data:
                normalized_note["chord_symbol"] = self._normalize_chord(
                    note_data["chord_symbol"]
                )

            normalized.append(normalized_note)

        return normalized

    def _normalize_effect(self, effect: str) -> dict:
        """Normalize a single effect notation to GP5 format."""
        detection = self.detector.detect(effect)

        gp5_effect = {
            "type": detection.notation_type,
            "original": effect,
            "confidence": detection.confidence,
        }

        # Map to GP5 specific values
        effect_mapping = {
            "bend": {"effect_name": "bend", "gp5_attr": "bend"},
            "slide_up": {"effect_name": "slide", "gp5_attr": "slides"},
            "slide_down": {"effect_name": "slide", "gp5_attr": "slides"},
            "hammer_on": {"effect_name": "hammer", "gp5_attr": "hammer"},
            "pull_off": {"effect_name": "hammer", "gp5_attr": "hammer"},
            "vibrato": {"effect_name": "vibrato", "gp5_attr": "vibrato"},
            "palm_mute": {"effect_name": "palmMute", "gp5_attr": "palmMute"},
            "ghost_note": {"effect_name": "ghostNote", "gp5_attr": "ghostNote"},
            "dead_note": {"effect_name": "deadNote", "gp5_attr": "type"},
        }

        if detection.notation_type in effect_mapping:
            gp5_effect.update(effect_mapping[detection.notation_type])

        return gp5_effect

    def _normalize_chord(self, chord_symbol: str) -> dict:
        """Normalize chord symbol to MusicXML/GP5 format."""
        detection = self.detector.detect(chord_symbol)

        return {
            "type": detection.notation_type,
            "original": chord_symbol,
            "confidence": detection.confidence,
        }
