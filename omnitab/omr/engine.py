"""OMR Engine using Audiveris or oemer."""

import subprocess
from pathlib import Path
from typing import Optional

from music21 import converter, note, chord


class OMREngine:
    """
    Optical Music Recognition engine.

    Supports:
    - Audiveris (Java, requires separate installation)
    - oemer (Python, included in requirements)
    """

    def __init__(self, backend: str = "oemer"):
        """
        Initialize OMR engine.

        Args:
            backend: "audiveris" or "oemer"
        """
        self.backend = backend

    def process(self, pdf_path: Path) -> Optional[Path]:
        """
        Process PDF and extract MusicXML.

        Args:
            pdf_path: Path to input PDF

        Returns:
            Path to generated MusicXML file, or None on failure
        """
        if self.backend == "audiveris":
            return self._process_audiveris(pdf_path)
        else:
            return self._process_oemer(pdf_path)

    def _process_audiveris(self, pdf_path: Path) -> Optional[Path]:
        """Process using Audiveris (requires Java)."""
        output_path = pdf_path.with_suffix(".musicxml")
        try:
            cmd = [
                "audiveris",
                "-batch",
                "-output",
                str(output_path),
                str(pdf_path),
            ]
            subprocess.run(cmd, timeout=300, check=True)
            return output_path if output_path.exists() else None
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _process_oemer(self, pdf_path: Path) -> Optional[Path]:
        """Process using oemer (Python OMR)."""
        from pdf2image import convert_from_path
        import oemer

        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            if not images:
                return None

            # Process first page (TODO: handle multi-page)
            img_path = pdf_path.with_suffix(".png")
            images[0].save(str(img_path), "PNG")

            # Run oemer
            output_path = pdf_path.with_suffix(".musicxml")
            oemer.generate(str(img_path), str(output_path))

            return output_path if output_path.exists() else None
        except Exception:
            return None

    def parse_musicxml(self, musicxml_path: Path) -> list[dict]:
        """
        Parse MusicXML to internal note representation.

        Args:
            musicxml_path: Path to MusicXML file

        Returns:
            List of note dictionaries
        """
        score = converter.parse(str(musicxml_path))
        notes_data = []

        for element in score.flatten().notesAndRests:
            if isinstance(element, note.Note):
                notes_data.append(
                    {
                        "type": "note",
                        "pitch": element.pitch.midi,
                        "name": element.pitch.nameWithOctave,
                        "duration": element.quarterLength,
                        "offset": element.offset,
                        "effects": [],
                    }
                )
            elif isinstance(element, chord.Chord):
                notes_data.append(
                    {
                        "type": "chord",
                        "pitches": [p.midi for p in element.pitches],
                        "names": [p.nameWithOctave for p in element.pitches],
                        "duration": element.quarterLength,
                        "offset": element.offset,
                        "effects": [],
                    }
                )
            elif isinstance(element, note.Rest):
                notes_data.append(
                    {
                        "type": "rest",
                        "duration": element.quarterLength,
                        "offset": element.offset,
                    }
                )

        return notes_data
