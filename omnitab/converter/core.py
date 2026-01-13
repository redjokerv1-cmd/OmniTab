"""Core converter implementation."""

from pathlib import Path

from omnitab.converter.result import ConversionResult
from omnitab.omr import OMREngine
from omnitab.notation import NotationNormalizer
from omnitab.gp5 import GP5Writer


class OmniTabConverter:
    """Main converter class that orchestrates the pipeline."""

    def __init__(self):
        self.omr_engine = OMREngine()
        self.notation_normalizer = NotationNormalizer()
        self.gp5_writer = GP5Writer()

    def convert(self, input_path: Path, output_path: Path) -> ConversionResult:
        """
        Convert PDF sheet music to Guitar Pro 5 format.

        Pipeline:
        1. OMR: PDF -> MusicXML
        2. Parse: MusicXML -> Note data
        3. Normalize: Various notations -> Standard format
        4. Generate: Note data -> GP5 file

        Args:
            input_path: Path to input PDF file
            output_path: Path to output GP5 file

        Returns:
            ConversionResult with success status and details
        """
        try:
            # Step 1: OMR - Extract music notation from PDF
            music_xml = self.omr_engine.process(input_path)
            if not music_xml:
                return ConversionResult(
                    success=False,
                    error="OMR failed to extract music notation from PDF",
                )

            # Step 2: Parse MusicXML to internal representation
            notes_data = self.omr_engine.parse_musicxml(music_xml)

            # Step 3: Normalize notation
            normalized_data = self.notation_normalizer.normalize(notes_data)

            # Step 4: Generate GP5 file
            self.gp5_writer.write(normalized_data, output_path)

            return ConversionResult(success=True, output_path=output_path)

        except Exception as e:
            return ConversionResult(success=False, error=str(e))
