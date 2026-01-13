"""
Header Detector - Extract tuning and capo from TAB sheet header

Detects:
1. Tuning notation: "①=E ②=C ③=G ④=D ⑤=G ⑥=C" or "Tuning: DADGAD"
2. Capo position: "Capo. fret 2" or "Capo 3"
3. Tempo: "♩= 65" or "BPM: 120"
4. Time signature: "4/4", "3/4"
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
except ImportError:
    easyocr = None


# Standard tuning reference
STANDARD_TUNING = ['E', 'B', 'G', 'D', 'A', 'E']

# Note name patterns
NOTE_PATTERN = r'[A-Ga-g][#b♯♭]?'


@dataclass
class HeaderInfo:
    """Extracted header information"""
    tuning: List[str] = None
    capo: int = 0
    tempo: int = 0
    time_signature: str = ""
    title: str = ""
    artist: str = ""
    
    def __post_init__(self):
        if self.tuning is None:
            self.tuning = STANDARD_TUNING.copy()
    
    @property
    def is_standard_tuning(self) -> bool:
        return self.tuning == STANDARD_TUNING
    
    @property
    def is_partial_tuning(self) -> bool:
        """True if tuning detection was incomplete"""
        return '?' in self.tuning
    
    @property
    def needs_manual_tuning(self) -> bool:
        """True if manual tuning input is recommended"""
        return self.is_partial_tuning or (not self.is_standard_tuning)
    
    def to_dict(self) -> dict:
        return {
            'tuning': self.tuning,
            'capo': self.capo,
            'tempo': self.tempo,
            'time_signature': self.time_signature,
            'title': self.title,
            'artist': self.artist,
            'is_standard_tuning': self.is_standard_tuning
        }


class HeaderDetector:
    """
    Detect tuning, capo, and other metadata from TAB sheet header.
    
    Scans the top portion of the image for text patterns.
    """
    
    # Regex patterns
    PATTERNS = {
        # Tuning: "①=E ②=C" or "1=E 2=C" style
        'tuning_numbered': re.compile(r'[①②③④⑤⑥1-6]\s*=\s*([A-Ga-g][#b♯♭]?)', re.IGNORECASE),
        
        # Tuning: "Tuning: DADGAD" or "Tuning: D A D G A D"
        'tuning_named': re.compile(r'[Tt]uning[:\s]+([A-Ga-g#b♯♭\s]{6,})', re.IGNORECASE),
        
        # Capo: "Capo fret 2", "Capo. 3", "Capo 2nd"
        'capo': re.compile(r'[Cc]apo\.?\s*(?:fret\s*)?(\d+)', re.IGNORECASE),
        
        # Tempo: "♩= 65", "BPM: 120", "Tempo: 90"
        'tempo': re.compile(r'(?:♩\s*=\s*|[Bb][Pp][Mm][:\s]*|[Tt]empo[:\s]*)(\d{2,3})'),
        
        # Time signature: "4/4", "3/4", "6/8"
        'time_sig': re.compile(r'\b(\d)/(\d)\b'),
    }
    
    def __init__(self, 
                 header_ratio: float = 0.25,
                 use_gpu: bool = False):
        """
        Args:
            header_ratio: Portion of image height to scan (0.25 = top 25%)
            use_gpu: Use GPU for OCR
        """
        self.header_ratio = header_ratio
        self.use_gpu = use_gpu
        self._reader = None
        
        if cv2 is None:
            raise ImportError("OpenCV required")
    
    @property
    def reader(self):
        if self._reader is None:
            if easyocr is None:
                raise ImportError("EasyOCR required")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        return self._reader
    
    def detect(self, image: np.ndarray) -> HeaderInfo:
        """
        Detect header information from image.
        
        Args:
            image: Full TAB sheet image
            
        Returns:
            HeaderInfo with detected values
        """
        # Extract header region (top portion)
        height = image.shape[0]
        header_height = int(height * self.header_ratio)
        header_region = image[:header_height]
        
        # OCR the header
        text = self._ocr_region(header_region)
        
        # Parse detected text
        info = HeaderInfo()
        
        # Detect tuning
        info.tuning = self._detect_tuning(text)
        
        # Detect capo
        info.capo = self._detect_capo(text)
        
        # Detect tempo
        info.tempo = self._detect_tempo(text)
        
        # Detect time signature
        info.time_signature = self._detect_time_sig(text)
        
        return info
    
    def _ocr_region(self, region: np.ndarray) -> str:
        """OCR a region and return combined text"""
        results = self.reader.readtext(region)
        
        # Combine all text
        texts = [r[1] for r in results]
        return ' '.join(texts)
    
    def _detect_tuning(self, text: str) -> List[str]:
        """Detect tuning from text"""
        tuning = [None] * 6
        
        # Try numbered format first: "①=E ②=C ③=G"
        numbered_matches = re.findall(
            r'([①②③④⑤⑥1-6])\s*=\s*([A-Ga-g][#b♯♭]?)', 
            text, re.IGNORECASE
        )
        
        if numbered_matches:
            for num_str, note in numbered_matches:
                # Convert to 0-based index
                if num_str in '①②③④⑤⑥':
                    idx = '①②③④⑤⑥'.index(num_str)
                else:
                    idx = int(num_str) - 1
                
                if 0 <= idx < 6:
                    tuning[idx] = note.upper()
            
            # If we got all 6, return them
            if all(t is not None for t in tuning):
                return tuning
        
        # Try "=Note" pattern (when numbers are not recognized)
        # e.g., OCR returns "=E =C =G =D =G =C" separately
        equal_notes = re.findall(r'=\s*([A-Ga-g][#b♯♭]?)', text, re.IGNORECASE)
        if len(equal_notes) >= 6:
            # Take first 6 notes in order
            return [n.upper() for n in equal_notes[:6]]
        elif len(equal_notes) >= 3:
            # Partial detection - check if it's non-standard
            notes = [n.upper() for n in equal_notes]
            # If any note is not in standard tuning, mark as alternate
            non_standard = any(n not in ['E', 'B', 'G', 'D', 'A'] for n in notes)
            if non_standard:
                # Return partial with marker (will be flagged as needing manual input)
                return notes + ['?'] * (6 - len(notes))  # e.g., ['E', 'C', 'G', '?', '?', '?']
        
        # Try named format: "Tuning: DADGAD"
        match = self.PATTERNS['tuning_named'].search(text)
        if match:
            tuning_str = match.group(1).strip()
            # Remove spaces and extract notes
            notes = re.findall(r'[A-Ga-g][#b♯♭]?', tuning_str)
            if len(notes) == 6:
                return [n.upper() for n in notes]
        
        # Default to standard tuning
        return STANDARD_TUNING.copy()
    
    def _detect_capo(self, text: str) -> int:
        """Detect capo position"""
        match = self.PATTERNS['capo'].search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 0
    
    def _detect_tempo(self, text: str) -> int:
        """Detect tempo (BPM)"""
        match = self.PATTERNS['tempo'].search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 0
    
    def _detect_time_sig(self, text: str) -> str:
        """Detect time signature"""
        match = self.PATTERNS['time_sig'].search(text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return ""
    
    def detect_file(self, path: str) -> HeaderInfo:
        """Detect from file path"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read: {path}")
        return self.detect(image)


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python header_detector.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Detecting header: {path}")
    print("=" * 50)
    
    detector = HeaderDetector()
    info = detector.detect_file(path)
    
    print(f"Tuning: {info.tuning}")
    print(f"  Standard: {info.is_standard_tuning}")
    print(f"  Partial: {info.is_partial_tuning}")
    print(f"  Needs manual: {info.needs_manual_tuning}")
    print(f"Capo: {info.capo}")
    print(f"Tempo: {info.tempo}")
    print(f"Time Sig: {info.time_signature}")
    
    if info.needs_manual_tuning:
        print()
        print("[!] Alternate tuning detected! Manual verification recommended")
        print("    Yellow Jacket tuning: 1=E 2=C 3=G 4=D 5=G 6=C")
