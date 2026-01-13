"""Notation type detector with confidence scoring."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Result of notation detection."""

    notation_type: str
    confidence: float
    original: str
    matched_pattern: Optional[str] = None


class NotationDetector:
    """
    Detects various notation types and calculates confidence.

    Confidence thresholds:
    - 0.85+: Auto-process
    - 0.7-0.85: AI-assisted
    - <0.7: Manual selection
    """

    PATTERNS = {
        # Chord notation patterns
        "chord_major": (r"^[A-G](#|b)?$", 0.95),
        "chord_minor": (r"^[A-G](#|b)?m$", 0.95),
        "chord_major7": (r"^[A-G](#|b)?(maj7|M7|△7)$", 0.92),
        "chord_minor7": (r"^[A-G](#|b)?(m7|-7)$", 0.92),
        "chord_dom7": (r"^[A-G](#|b)?7$", 0.94),
        "chord_dim": (r"^[A-G](#|b)?(dim|o|°)$", 0.90),
        "chord_aug": (r"^[A-G](#|b)?(\+|aug)$", 0.90),
        "chord_sus": (r"^[A-G](#|b)?sus[24]$", 0.93),
        # Effect notation patterns
        "bend": (r"^(\d+)[bB^](\d+)$", 0.88),
        "slide_up": (r"^(\d+)[/sS](\d+)$", 0.85),
        "slide_down": (r"^(\d+)[\\](\d+)$", 0.85),
        "hammer_on": (r"^(\d+)[hH](\d+)$", 0.90),
        "pull_off": (r"^(\d+)[pP](\d+)$", 0.90),
        "vibrato": (r"^(\d+)[~]$", 0.88),
        "dead_note": (r"^[xX]$", 0.95),
        "ghost_note": (r"^\((\d+)\)$", 0.92),
        "palm_mute": (r"^PM\.?$", 0.93),
        # Harmonic patterns
        "natural_harmonic": (r"^<(\d+)>$", 0.90),
        "artificial_harmonic": (r"^(\d+)\[AH\]$", 0.88),
    }

    def detect(self, notation: str) -> DetectionResult:
        """
        Detect notation type and calculate confidence.

        Args:
            notation: Input notation string

        Returns:
            DetectionResult with type, confidence, and original
        """
        best_match = None
        best_confidence = 0.0
        best_pattern = None

        for notation_type, (pattern, base_confidence) in self.PATTERNS.items():
            if re.match(pattern, notation.strip()):
                if base_confidence > best_confidence:
                    best_confidence = base_confidence
                    best_match = notation_type
                    best_pattern = pattern

        if best_match:
            return DetectionResult(
                notation_type=best_match,
                confidence=best_confidence,
                original=notation,
                matched_pattern=best_pattern,
            )

        return DetectionResult(
            notation_type="unknown",
            confidence=0.0,
            original=notation,
        )

    def is_auto_processable(self, result: DetectionResult) -> bool:
        """Check if result has high enough confidence for auto-processing."""
        return result.confidence >= 0.85

    def needs_ai_assistance(self, result: DetectionResult) -> bool:
        """Check if result needs AI assistance."""
        return 0.7 <= result.confidence < 0.85

    def needs_manual_selection(self, result: DetectionResult) -> bool:
        """Check if result needs manual user selection."""
        return result.confidence < 0.7
