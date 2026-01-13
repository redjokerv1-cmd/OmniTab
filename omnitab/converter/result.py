"""Conversion result data class."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ConversionResult:
    """Result of PDF to GP5 conversion."""

    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
