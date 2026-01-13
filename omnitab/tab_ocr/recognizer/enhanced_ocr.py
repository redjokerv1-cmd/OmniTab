"""
Enhanced TAB OCR - Combines line removal with optimized parameters

Best configuration found through testing:
- Staff line removal: kernel=40, repair=2
- Chord grouping: x_threshold=5
- Result: 0 problems, max 5 notes per chord
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import easyocr
except ImportError:
    easyocr = None

# Import header detector
try:
    from .header_detector import HeaderDetector, HeaderInfo
except ImportError:
    HeaderDetector = None
    HeaderInfo = None


@dataclass
class RecognizedDigit:
    """A recognized digit from TAB"""
    value: int
    x: float
    y: float
    width: int
    height: int
    confidence: float
    string_num: Optional[int] = None  # 1-6, assigned later


@dataclass
class TabChord:
    """A chord (group of simultaneous notes)"""
    notes: List[RecognizedDigit] = field(default_factory=list)
    x_position: float = 0.0
    
    @property
    def frets(self) -> List[int]:
        """Fret values sorted by string (top=1 to bottom=6)"""
        return [n.value for n in sorted(self.notes, key=lambda n: n.y)]
    
    @property
    def confidence(self) -> float:
        return sum(n.confidence for n in self.notes) / max(len(self.notes), 1)


@dataclass
class TabSystem:
    """A TAB system (one line of 6-string tablature)"""
    digits: List[RecognizedDigit] = field(default_factory=list)
    chords: List[TabChord] = field(default_factory=list)
    y_start: float = 0.0
    y_end: float = 0.0


class EnhancedTabOCR:
    """
    Enhanced TAB OCR with optimal parameters.
    
    Pipeline:
    1. Remove staff lines (prevents digit connection)
    2. Binary threshold + contour detection
    3. Size filtering + scaling
    4. Individual digit OCR
    5. Tight chord grouping (x_threshold=5)
    """
    
    # Optimal parameters from testing
    DEFAULT_LINE_KERNEL = 40
    DEFAULT_REPAIR_KERNEL = 2
    DEFAULT_X_THRESHOLD = 5
    DEFAULT_Y_GAP_THRESHOLD = 80
    
    def __init__(self,
                 # Line removal
                 line_kernel: int = DEFAULT_LINE_KERNEL,
                 repair_kernel: int = DEFAULT_REPAIR_KERNEL,
                 # OCR
                 binary_threshold: int = 200,
                 min_digit_width: int = 3,
                 max_digit_width: int = 40,
                 min_digit_height: int = 5,
                 max_digit_height: int = 35,
                 min_confidence: float = 0.2,
                 scale_factor: int = 3,
                 # Grouping
                 x_threshold: float = DEFAULT_X_THRESHOLD,
                 y_gap_threshold: float = DEFAULT_Y_GAP_THRESHOLD,
                 # Hardware
                 use_gpu: bool = False):
        
        self.line_kernel = line_kernel
        self.repair_kernel = repair_kernel
        self.binary_threshold = binary_threshold
        self.min_digit_width = min_digit_width
        self.max_digit_width = max_digit_width
        self.min_digit_height = min_digit_height
        self.max_digit_height = max_digit_height
        self.min_confidence = min_confidence
        self.scale_factor = scale_factor
        self.x_threshold = x_threshold
        self.y_gap_threshold = y_gap_threshold
        self.use_gpu = use_gpu
        self.detect_header = True  # Auto-detect tuning/capo
        
        self._reader = None
        self._header_detector = None
        
        if cv2 is None:
            raise ImportError("OpenCV required: pip install opencv-python")
    
    @property
    def header_detector(self):
        if self._header_detector is None and HeaderDetector is not None:
            self._header_detector = HeaderDetector(use_gpu=self.use_gpu)
        return self._header_detector
    
    @property
    def reader(self):
        if self._reader is None:
            if easyocr is None:
                raise ImportError("EasyOCR required: pip install easyocr")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        return self._reader
    
    def process(self, image: np.ndarray, tab_systems: List = None) -> Dict:
        """
        Process TAB image with optimal pipeline.
        
        Returns:
            Dict with:
            - 'digits': All recognized digits
            - 'systems': TAB systems with chords
            - 'stats': Processing statistics
        """
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Remove staff lines
        clean_gray = self._remove_lines(gray)
        
        # Step 3: Binary threshold
        _, binary = cv2.threshold(clean_gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        binary_inv = cv2.bitwise_not(binary)
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Filter candidates
        candidates = self._filter_candidates(contours)
        
        # Step 6: OCR each candidate
        digits = self._recognize_digits(clean_gray, candidates)
        
        # Step 7: Group into systems
        systems = self._group_into_systems(digits)
        
        # Step 8: Group into chords (tight threshold)
        for system in systems:
            system.chords = self._group_into_chords(system.digits)
        
        # Stats
        total_chords = sum(len(s.chords) for s in systems)
        avg_conf = sum(d.confidence for d in digits) / max(len(digits), 1)
        problems = sum(1 for s in systems for c in s.chords if len(c.notes) > 6)
        
        # Detect header (tuning, capo)
        header_info = None
        if self.detect_header and self.header_detector:
            try:
                header_info = self.header_detector.detect(image)
            except Exception:
                pass
        
        result = {
            'digits': digits,
            'systems': systems,
            'stats': {
                'total_digits': len(digits),
                'total_systems': len(systems),
                'total_chords': total_chords,
                'avg_confidence': avg_conf,
                'problems': problems
            }
        }
        
        if header_info:
            result['header'] = {
                'tuning': header_info.tuning,
                'capo': header_info.capo,
                'tempo': header_info.tempo,
                'time_signature': header_info.time_signature,
                'is_standard_tuning': header_info.is_standard_tuning,
                'needs_manual_tuning': header_info.needs_manual_tuning
            }
        
        return result
    
    def _remove_lines(self, gray: np.ndarray) -> np.ndarray:
        """Remove horizontal staff lines"""
        # Binary threshold (invert)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal kernel
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_kernel, 1))
        
        # Detect lines
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        # Remove lines
        result = gray.copy()
        result[lines == 255] = 255
        
        # Repair digits
        if self.repair_kernel > 0:
            repair_k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.repair_kernel, self.repair_kernel)
            )
            result_inv = cv2.bitwise_not(result)
            result_inv = cv2.dilate(result_inv, repair_k, iterations=1)
            result = cv2.bitwise_not(result_inv)
        
        return result
    
    def _filter_candidates(self, contours) -> List[tuple]:
        """Filter contours to digit candidates"""
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if not (self.min_digit_width <= w <= self.max_digit_width):
                continue
            if not (self.min_digit_height <= h <= self.max_digit_height):
                continue
            
            aspect = w / h if h > 0 else 0
            if not (0.2 <= aspect <= 2.0):
                continue
            
            candidates.append((x, y, w, h))
        
        return candidates
    
    def _recognize_digits(self, gray: np.ndarray, candidates: List[tuple]) -> List[RecognizedDigit]:
        """OCR each digit"""
        digits = []
        
        for x, y, w, h in candidates:
            # Extract with padding
            pad = 5
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad)
            roi = gray[y1:y2, x1:x2]
            
            # Scale up
            if self.scale_factor > 1:
                roi = cv2.resize(roi, None, fx=self.scale_factor, fy=self.scale_factor,
                                interpolation=cv2.INTER_CUBIC)
            
            # Add border
            roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            
            # OCR
            results = self.reader.readtext(roi, allowlist='0123456789', paragraph=False)
            
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
    
    def _group_into_systems(self, digits: List[RecognizedDigit]) -> List[TabSystem]:
        """Group by Y position"""
        if not digits:
            return []
        
        sorted_d = sorted(digits, key=lambda d: d.y)
        systems = []
        current = [sorted_d[0]]
        
        for d in sorted_d[1:]:
            if d.y - current[-1].y > self.y_gap_threshold:
                systems.append(TabSystem(
                    digits=current,
                    y_start=min(n.y for n in current),
                    y_end=max(n.y for n in current)
                ))
                current = [d]
            else:
                current.append(d)
        
        if current:
            systems.append(TabSystem(
                digits=current,
                y_start=min(n.y for n in current),
                y_end=max(n.y for n in current)
            ))
        
        return systems
    
    def _group_into_chords(self, digits: List[RecognizedDigit]) -> List[TabChord]:
        """Group by X position with tight threshold"""
        if not digits:
            return []
        
        sorted_d = sorted(digits, key=lambda d: d.x)
        chords = []
        current = [sorted_d[0]]
        
        for d in sorted_d[1:]:
            if d.x - current[-1].x < self.x_threshold:
                current.append(d)
            else:
                chords.append(TabChord(
                    notes=current,
                    x_position=sum(n.x for n in current) / len(current)
                ))
                current = [d]
        
        if current:
            chords.append(TabChord(
                notes=current,
                x_position=sum(n.x for n in current) / len(current)
            ))
        
        return chords
    
    def process_file(self, path: str) -> Dict:
        """Process image file"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read: {path}")
        return self.process(image)


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_ocr.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Processing: {path}")
    print("="*60)
    
    ocr = EnhancedTabOCR()
    result = ocr.process_file(path)
    
    stats = result['stats']
    print(f"Digits: {stats['total_digits']}")
    print(f"Systems: {stats['total_systems']}")
    print(f"Chords: {stats['total_chords']}")
    print(f"Confidence: {stats['avg_confidence']:.1%}")
    print(f"Problems: {stats['problems']}")
    
    if 'header' in result:
        h = result['header']
        print()
        print("Header Info:")
        print(f"  Tuning: {h['tuning']}")
        print(f"  Capo: {h['capo']}")
        print(f"  Standard: {h['is_standard_tuning']}")
        if h['needs_manual_tuning']:
            print("  [!] Alternate tuning - manual verification recommended")
    print()
    
    for i, system in enumerate(result['systems']):
        print(f"System {i+1}: {len(system.chords)} chords")
        for j, chord in enumerate(system.chords[:5]):
            print(f"  [{j+1}] frets={chord.frets} ({chord.confidence:.0%})")
        if len(system.chords) > 5:
            print(f"  ... +{len(system.chords)-5} more")
        print()
