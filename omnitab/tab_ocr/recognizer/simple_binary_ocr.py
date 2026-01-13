"""
Simple Binary OCR - Optimized for black/white sheet music

Key insight: Sheet music is almost always black and white.
Simple binary thresholding is more effective than complex preprocessing.

Improvements over complex preprocessing:
- Fewer false positives (cleaner contours)
- Faster processing
- Better digit separation
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
except ImportError:
    easyocr = None


@dataclass
class RecognizedDigit:
    """A recognized digit from TAB"""
    value: int
    x: float
    y: float
    width: int
    height: int
    confidence: float


@dataclass 
class TabChordData:
    """A chord (group of simultaneous notes)"""
    notes: List[RecognizedDigit]
    x_position: float
    
    @property
    def values(self) -> List[int]:
        """Get fret values sorted by string (top to bottom)"""
        return [n.value for n in sorted(self.notes, key=lambda n: n.y)]
    
    @property
    def avg_confidence(self) -> float:
        return sum(n.confidence for n in self.notes) / max(len(self.notes), 1)


@dataclass
class TabSystemData:
    """A TAB system (one line of tablature)"""
    digits: List[RecognizedDigit]
    y_start: float
    y_end: float
    chords: List[TabChordData] = None
    
    def __post_init__(self):
        if self.chords is None:
            self.chords = []


class SimpleBinaryOCR:
    """
    Simple binary OCR optimized for black/white sheet music.
    
    Pipeline:
    1. Binary threshold (B&W)
    2. Contour detection
    3. Size filtering
    4. Individual digit OCR
    5. Position-based grouping
    """
    
    def __init__(self,
                 threshold: int = 200,
                 min_digit_width: int = 3,
                 max_digit_width: int = 40,
                 min_digit_height: int = 5,
                 max_digit_height: int = 35,
                 min_confidence: float = 0.2,
                 scale_factor: int = 3,
                 use_gpu: bool = False):
        """
        Initialize SimpleBinaryOCR.
        
        Args:
            threshold: Binary threshold value (pixels > threshold = white)
            min/max_digit_width: Width constraints for digit candidates
            min/max_digit_height: Height constraints for digit candidates
            min_confidence: Minimum OCR confidence to accept
            scale_factor: Scale factor for small digits (3 = 3x larger)
            use_gpu: Use GPU for OCR
        """
        self.threshold = threshold
        self.min_digit_width = min_digit_width
        self.max_digit_width = max_digit_width
        self.min_digit_height = min_digit_height
        self.max_digit_height = max_digit_height
        self.min_confidence = min_confidence
        self.scale_factor = scale_factor
        self.use_gpu = use_gpu
        
        self._reader = None
        
        if cv2 is None:
            raise ImportError("OpenCV required: pip install opencv-python")
    
    @property
    def reader(self):
        """Lazy load OCR reader"""
        if self._reader is None:
            if easyocr is None:
                raise ImportError("EasyOCR required: pip install easyocr")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        return self._reader
    
    def process(self, image: np.ndarray) -> Dict:
        """
        Process a TAB image and extract all data.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Dictionary with:
            - 'digits': List of all recognized digits
            - 'systems': List of TAB systems
            - 'total_chords': Total chord count
            - 'avg_confidence': Average confidence
        """
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Simple binary threshold
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        binary_inv = cv2.bitwise_not(binary)
        
        # Step 3: Find contours
        contours, _ = cv2.findContours(
            binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Step 4: Filter by size
        candidates = self._filter_digit_candidates(contours)
        
        # Step 5: OCR each candidate
        digits = self._recognize_digits(gray, candidates)
        
        # Step 6: Group into TAB systems (by Y position)
        systems = self._group_into_systems(digits)
        
        # Step 7: Group into chords (by X position within each system)
        for system in systems:
            system.chords = self._group_into_chords(system.digits)
        
        # Calculate stats
        total_chords = sum(len(s.chords) for s in systems)
        avg_conf = (sum(d.confidence for d in digits) / max(len(digits), 1))
        
        return {
            'digits': digits,
            'systems': systems,
            'total_chords': total_chords,
            'avg_confidence': avg_conf
        }
    
    def _filter_digit_candidates(self, 
                                  contours: List) -> List[Tuple[int, int, int, int]]:
        """Filter contours to likely digit candidates"""
        candidates = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Size constraints
            if not (self.min_digit_width <= w <= self.max_digit_width):
                continue
            if not (self.min_digit_height <= h <= self.max_digit_height):
                continue
            
            # Aspect ratio (digits are roughly square or taller)
            aspect = w / h if h > 0 else 0
            if not (0.2 <= aspect <= 2.0):
                continue
            
            candidates.append((x, y, w, h))
        
        return candidates
    
    def _recognize_digits(self,
                          gray: np.ndarray,
                          candidates: List[Tuple[int, int, int, int]]) -> List[RecognizedDigit]:
        """OCR each digit candidate"""
        digits = []
        
        for x, y, w, h in candidates:
            # Extract ROI with padding
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            
            roi = gray[y1:y2, x1:x2]
            
            # Scale up small digits for better OCR
            if self.scale_factor > 1:
                roi = cv2.resize(
                    roi, None, 
                    fx=self.scale_factor, 
                    fy=self.scale_factor, 
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Add white border for better OCR
            roi = cv2.copyMakeBorder(
                roi, 10, 10, 10, 10, 
                cv2.BORDER_CONSTANT, value=255
            )
            
            # OCR with digit-only allowlist
            results = self.reader.readtext(
                roi, 
                allowlist='0123456789', 
                paragraph=False
            )
            
            if results:
                _, text, conf = results[0]
                try:
                    value = int(text.strip())
                    if 0 <= value <= 24 and conf >= self.min_confidence:
                        digits.append(RecognizedDigit(
                            value=value,
                            x=x + w/2,
                            y=y + h/2,
                            width=w,
                            height=h,
                            confidence=conf
                        ))
                except ValueError:
                    pass
        
        return digits
    
    def _group_into_systems(self,
                            digits: List[RecognizedDigit],
                            y_gap_threshold: float = 80) -> List[TabSystemData]:
        """Group digits into TAB systems by Y position"""
        if not digits:
            return []
        
        # Sort by Y
        sorted_digits = sorted(digits, key=lambda d: d.y)
        
        systems = []
        current_digits = [sorted_digits[0]]
        
        for digit in sorted_digits[1:]:
            if digit.y - current_digits[-1].y > y_gap_threshold:
                # New system
                systems.append(TabSystemData(
                    digits=current_digits,
                    y_start=min(d.y for d in current_digits),
                    y_end=max(d.y for d in current_digits)
                ))
                current_digits = [digit]
            else:
                current_digits.append(digit)
        
        # Don't forget last system
        if current_digits:
            systems.append(TabSystemData(
                digits=current_digits,
                y_start=min(d.y for d in current_digits),
                y_end=max(d.y for d in current_digits)
            ))
        
        return systems
    
    def _group_into_chords(self,
                           digits: List[RecognizedDigit],
                           x_threshold: float = 25) -> List[TabChordData]:
        """Group digits into chords by X position"""
        if not digits:
            return []
        
        # Sort by X
        sorted_digits = sorted(digits, key=lambda d: d.x)
        
        chords = []
        current_notes = [sorted_digits[0]]
        
        for digit in sorted_digits[1:]:
            if digit.x - current_notes[-1].x < x_threshold:
                current_notes.append(digit)
            else:
                chords.append(TabChordData(
                    notes=current_notes,
                    x_position=sum(n.x for n in current_notes) / len(current_notes)
                ))
                current_notes = [digit]
        
        # Don't forget last chord
        if current_notes:
            chords.append(TabChordData(
                notes=current_notes,
                x_position=sum(n.x for n in current_notes) / len(current_notes)
            ))
        
        return chords
    
    def process_file(self, file_path: str) -> Dict:
        """Process an image file"""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        return self.process(image)


# CLI for testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_binary_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Processing: {image_path}")
    print("=" * 60)
    
    ocr = SimpleBinaryOCR()
    result = ocr.process_file(image_path)
    
    print(f"Total digits recognized: {len(result['digits'])}")
    print(f"TAB systems found: {len(result['systems'])}")
    print(f"Total chords: {result['total_chords']}")
    print(f"Average confidence: {result['avg_confidence']:.2%}")
    print()
    
    for i, system in enumerate(result['systems']):
        print(f"System {i+1}: y={system.y_start:.0f}-{system.y_end:.0f}, "
              f"{len(system.digits)} digits, {len(system.chords)} chords")
        
        for j, chord in enumerate(system.chords[:5]):
            print(f"  Chord {j+1}: {chord.values} ({chord.avg_confidence:.0%})")
        
        if len(system.chords) > 5:
            print(f"  ... and {len(system.chords) - 5} more chords")
        print()
