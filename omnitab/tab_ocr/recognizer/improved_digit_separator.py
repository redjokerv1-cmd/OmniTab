"""
Improved Digit Separator - Contour-based digit recognition for TAB OCR

Phase 1 Implementation:
1. CLAHE enhancement
2. Staff line removal (6 horizontal lines)
3. Contour-based digit separation
4. Individual digit OCR
5. Position-based grouping

Target: 60% → 85% accuracy
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
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
class DigitCandidate:
    """A detected digit candidate from contour analysis"""
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray  # Cropped digit image
    
    # OCR results
    value: Optional[int] = None
    confidence: float = 0.0
    raw_text: str = ""
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def area(self) -> int:
        return self.width * self.height


class EnhancedPreprocessor:
    """Enhanced image preprocessing with CLAHE and line removal"""
    
    def __init__(self, 
                 clahe_clip_limit: float = 3.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        
        if cv2 is None:
            raise ImportError("OpenCV required")
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement for better contrast"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def remove_staff_lines(self, 
                           image: np.ndarray,
                           line_thickness: int = 2) -> np.ndarray:
        """
        Remove horizontal staff lines from TAB image.
        
        These lines interfere with digit recognition.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create horizontal kernel (long and thin)
        # This detects horizontal lines
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (max(40, gray.shape[1] // 20), 1)
        )
        
        # Detect horizontal lines
        detected_lines = cv2.morphologyEx(
            binary, 
            cv2.MORPH_OPEN, 
            horizontal_kernel, 
            iterations=2
        )
        
        # Dilate lines slightly to ensure complete removal
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (1, line_thickness + 1)
        )
        detected_lines = cv2.dilate(detected_lines, dilate_kernel, iterations=1)
        
        # Invert to get mask (lines are white, rest is black)
        # Then invert again for subtraction
        lines_mask = detected_lines
        
        # Remove lines from original (set line pixels to white)
        result = gray.copy()
        result[lines_mask > 0] = 255
        
        return result
    
    def full_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline"""
        # 1. CLAHE enhancement
        enhanced = self.enhance(image)
        
        # 2. Remove staff lines
        no_lines = self.remove_staff_lines(enhanced)
        
        # 3. Additional denoising
        denoised = cv2.fastNlMeansDenoising(no_lines, None, 10, 7, 21)
        
        return denoised


class DigitSeparator:
    """Separate individual digits using contour analysis"""
    
    def __init__(self,
                 min_digit_width: int = 5,
                 max_digit_width: int = 50,
                 min_digit_height: int = 8,
                 max_digit_height: int = 40,
                 min_aspect_ratio: float = 0.2,
                 max_aspect_ratio: float = 2.5,
                 padding: int = 3):
        """
        Initialize DigitSeparator.
        
        Args:
            min/max_digit_width: Width constraints for digit candidates
            min/max_digit_height: Height constraints for digit candidates
            min/max_aspect_ratio: Aspect ratio (w/h) constraints
            padding: Padding around extracted digits
        """
        self.min_digit_width = min_digit_width
        self.max_digit_width = max_digit_width
        self.min_digit_height = min_digit_height
        self.max_digit_height = max_digit_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.padding = padding
        
        if cv2 is None:
            raise ImportError("OpenCV required")
    
    def find_digit_candidates(self, 
                              preprocessed: np.ndarray,
                              original: np.ndarray = None) -> List[DigitCandidate]:
        """
        Find digit candidates using contour analysis.
        
        Args:
            preprocessed: Preprocessed grayscale image (lines removed)
            original: Original image for cropping (optional)
            
        Returns:
            List of DigitCandidate objects
        """
        if original is None:
            original = preprocessed
        
        # Convert original to grayscale if needed
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
        
        # Binary threshold
        _, binary = cv2.threshold(
            preprocessed, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        candidates = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if not self._is_valid_digit_size(w, h):
                continue
            
            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Extract digit image with padding
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(original_gray.shape[1], x + w + self.padding)
            y2 = min(original_gray.shape[0], y + h + self.padding)
            
            digit_img = original_gray[y1:y2, x1:x2].copy()
            
            # Add white border for OCR
            digit_img = cv2.copyMakeBorder(
                digit_img, 5, 5, 5, 5,
                cv2.BORDER_CONSTANT, value=255
            )
            
            candidates.append(DigitCandidate(
                x=x,
                y=y,
                width=w,
                height=h,
                image=digit_img
            ))
        
        # Sort by position (left to right, top to bottom)
        candidates.sort(key=lambda c: (c.y // 15, c.x))
        
        return candidates
    
    def _is_valid_digit_size(self, width: int, height: int) -> bool:
        """Check if size is valid for a digit"""
        return (self.min_digit_width <= width <= self.max_digit_width and
                self.min_digit_height <= height <= self.max_digit_height)
    
    def merge_adjacent_digits(self,
                              candidates: List[DigitCandidate],
                              x_threshold: int = 8,
                              y_threshold: int = 5) -> List[DigitCandidate]:
        """
        Merge horizontally adjacent digits (for 2-digit numbers like 10, 12).
        
        Args:
            candidates: List of digit candidates
            x_threshold: Max horizontal gap to merge
            y_threshold: Max vertical difference to consider same line
            
        Returns:
            Merged candidates
        """
        if len(candidates) < 2:
            return candidates
        
        merged = []
        skip_next = False
        
        for i, current in enumerate(candidates):
            if skip_next:
                skip_next = False
                continue
            
            if i + 1 < len(candidates):
                next_digit = candidates[i + 1]
                
                # Check if they should be merged
                x_gap = next_digit.x - (current.x + current.width)
                y_diff = abs(current.center_y - next_digit.center_y)
                
                if x_gap < x_threshold and y_diff < y_threshold:
                    # Merge: only if current is a single digit (1 or 2)
                    if current.value is not None and current.value < 3:
                        # Potential 2-digit number
                        # Don't merge yet, let OCR handle it
                        pass
            
            merged.append(current)
        
        return merged


class IndividualDigitOCR:
    """OCR for individual digit images"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._reader = None
        
        if easyocr is None:
            raise ImportError("EasyOCR required")
    
    @property
    def reader(self):
        """Lazy load EasyOCR reader"""
        if self._reader is None:
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        return self._reader
    
    def recognize_single(self, digit_image: np.ndarray) -> Tuple[Optional[int], float, str]:
        """
        Recognize a single digit image.
        
        Args:
            digit_image: Cropped digit image
            
        Returns:
            (value, confidence, raw_text)
        """
        # Run OCR with strict digit allowlist
        results = self.reader.readtext(
            digit_image,
            allowlist='0123456789',
            paragraph=False,
            detail=1
        )
        
        if not results:
            return None, 0.0, ""
        
        # Get best result
        best = max(results, key=lambda r: r[2])
        bbox, text, confidence = best
        
        # Parse to int
        text = text.strip()
        try:
            value = int(text)
            if 0 <= value <= 24:  # Valid fret range
                return value, confidence, text
            return None, confidence, text
        except ValueError:
            return None, confidence, text
    
    def recognize_batch(self, 
                        candidates: List[DigitCandidate],
                        min_confidence: float = 0.3) -> List[DigitCandidate]:
        """
        Recognize all digit candidates.
        
        Args:
            candidates: List of DigitCandidate objects
            min_confidence: Minimum confidence threshold
            
        Returns:
            Updated candidates with OCR results
        """
        recognized = []
        
        for candidate in candidates:
            value, confidence, raw_text = self.recognize_single(candidate.image)
            
            candidate.value = value
            candidate.confidence = confidence
            candidate.raw_text = raw_text
            
            if value is not None and confidence >= min_confidence:
                recognized.append(candidate)
        
        return recognized


class ImprovedChordGrouper:
    """Group digits into chords based on vertical position"""
    
    def __init__(self, 
                 x_threshold: float = 20,
                 y_row_threshold: float = 15):
        """
        Args:
            x_threshold: Max X distance to consider same chord
            y_row_threshold: Y distance to distinguish TAB rows
        """
        self.x_threshold = x_threshold
        self.y_row_threshold = y_row_threshold
    
    def group_into_chords(self, 
                          digits: List[DigitCandidate]) -> List[List[DigitCandidate]]:
        """
        Group digits into chords.
        
        Digits at similar X positions are part of the same chord.
        """
        if not digits:
            return []
        
        # Sort by X position
        sorted_digits = sorted(digits, key=lambda d: d.center_x)
        
        chords = []
        current_chord = [sorted_digits[0]]
        
        for digit in sorted_digits[1:]:
            last_x = current_chord[-1].center_x
            
            if digit.center_x - last_x < self.x_threshold:
                # Same chord
                current_chord.append(digit)
            else:
                # New chord
                chords.append(current_chord)
                current_chord = [digit]
        
        # Don't forget last chord
        if current_chord:
            chords.append(current_chord)
        
        return chords
    
    def chord_to_tab_format(self, 
                            chord: List[DigitCandidate],
                            num_strings: int = 6) -> List[Optional[int]]:
        """
        Convert chord digits to TAB format [S1, S2, S3, S4, S5, S6].
        
        Assigns each digit to a string based on Y position.
        """
        if not chord:
            return [None] * num_strings
        
        # Sort by Y (top to bottom = string 1 to 6)
        sorted_by_y = sorted(chord, key=lambda d: d.center_y)
        
        # Find Y range
        y_min = sorted_by_y[0].center_y
        y_max = sorted_by_y[-1].center_y
        y_range = max(y_max - y_min, 1)
        
        # Assign each digit to a string
        result = [None] * num_strings
        
        for digit in sorted_by_y:
            # Calculate relative position (0.0 = top, 1.0 = bottom)
            relative_y = (digit.center_y - y_min) / y_range if y_range > 1 else 0.5
            
            # Map to string (0 = string 1, 5 = string 6)
            string_idx = int(relative_y * (num_strings - 1) + 0.5)
            string_idx = max(0, min(num_strings - 1, string_idx))
            
            # Don't overwrite existing
            if result[string_idx] is None:
                result[string_idx] = digit.value
            else:
                # Find nearest empty slot
                for offset in range(1, num_strings):
                    for direction in [-1, 1]:
                        new_idx = string_idx + offset * direction
                        if 0 <= new_idx < num_strings and result[new_idx] is None:
                            result[new_idx] = digit.value
                            break
                    else:
                        continue
                    break
        
        return result


class ImprovedTabOCREngine:
    """
    Improved TAB OCR Engine using contour-based digit separation.
    
    Phase 1 implementation targeting 85% accuracy.
    """
    
    def __init__(self, use_gpu: bool = False):
        self.preprocessor = EnhancedPreprocessor()
        self.separator = DigitSeparator()
        self.ocr = IndividualDigitOCR(use_gpu=use_gpu)
        self.grouper = ImprovedChordGrouper()
    
    def process_image(self, 
                      image: np.ndarray,
                      debug: bool = False) -> dict:
        """
        Process a TAB image and extract chord data.
        
        Args:
            image: Input image (BGR or grayscale)
            debug: If True, return debug information
            
        Returns:
            Dictionary with:
            - 'chords': List of chord arrays [[S1, S2, ...], ...]
            - 'confidence': Overall confidence
            - 'digit_count': Number of recognized digits
            - 'debug': Debug info (if requested)
        """
        # Step 1: Preprocess
        preprocessed = self.preprocessor.full_preprocess(image)
        
        # Step 2: Find digit candidates
        candidates = self.separator.find_digit_candidates(preprocessed, image)
        
        # Step 3: OCR each candidate
        recognized = self.ocr.recognize_batch(candidates, min_confidence=0.3)
        
        # Step 4: Group into chords
        chord_groups = self.grouper.group_into_chords(recognized)
        
        # Step 5: Convert to TAB format
        chords = []
        for group in chord_groups:
            tab_chord = self.grouper.chord_to_tab_format(group)
            chords.append(tab_chord)
        
        # Calculate confidence
        if recognized:
            avg_confidence = sum(d.confidence for d in recognized) / len(recognized)
        else:
            avg_confidence = 0.0
        
        result = {
            'chords': chords,
            'confidence': avg_confidence,
            'digit_count': len(recognized),
            'chord_count': len(chords)
        }
        
        if debug:
            result['debug'] = {
                'preprocessed': preprocessed,
                'candidates_found': len(candidates),
                'candidates_recognized': len(recognized),
                'digits': [(d.value, d.center_x, d.center_y, d.confidence) 
                          for d in recognized]
            }
        
        return result
    
    def process_file(self, 
                     file_path: str,
                     debug: bool = False) -> dict:
        """
        Process a TAB image file.
        
        Args:
            file_path: Path to image file
            debug: If True, return debug information
            
        Returns:
            Same as process_image()
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        
        return self.process_image(image, debug=debug)


# CLI for testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improved_digit_separator.py <image_path>")
        print("Example: python improved_digit_separator.py yellow_jacket.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug_mode = '--debug' in sys.argv
    
    print(f"Processing: {image_path}")
    print("=" * 60)
    
    engine = ImprovedTabOCREngine(use_gpu=False)
    result = engine.process_file(image_path, debug=debug_mode)
    
    print(f"Digits recognized: {result['digit_count']}")
    print(f"Chords found: {result['chord_count']}")
    print(f"Average confidence: {result['confidence']:.2%}")
    print()
    
    print("Chords (TAB format - S1 to S6):")
    print("-" * 40)
    for i, chord in enumerate(result['chords'][:20]):
        # Format: show only non-None values
        formatted = []
        for s, fret in enumerate(chord, 1):
            if fret is not None:
                formatted.append(f"S{s}={fret}")
        if formatted:
            print(f"  {i+1:2d}. [{', '.join(formatted)}]")
    
    if len(result['chords']) > 20:
        print(f"  ... and {len(result['chords']) - 20} more chords")
    
    if debug_mode and 'debug' in result:
        print()
        print("Debug info:")
        print(f"  Candidates found: {result['debug']['candidates_found']}")
        print(f"  Candidates recognized: {result['debug']['candidates_recognized']}")
        
        # Save preprocessed image
        cv2.imwrite('debug_preprocessed.png', result['debug']['preprocessed'])
        print("  Saved: debug_preprocessed.png")
