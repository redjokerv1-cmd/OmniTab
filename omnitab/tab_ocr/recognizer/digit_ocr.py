"""DigitOCR - Recognize fret numbers (0-24) in TAB"""

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
class DetectedDigit:
    """A detected digit/number with position"""
    value: int              # The fret number (0-24)
    x: float               # X center position
    y: float               # Y center position
    width: float           # Bounding box width
    height: float          # Bounding box height
    confidence: float      # OCR confidence (0-1)
    raw_text: str          # Original OCR text
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2)"""
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2
        )


class DigitOCR:
    """
    Recognize fret numbers in guitar TAB images.
    
    Handles:
    - Single digits (0-9)
    - Double digits (10-24)
    - Special cases (parentheses around numbers)
    """
    
    def __init__(self, 
                 use_gpu: bool = False,
                 min_confidence: float = 0.3):
        """
        Initialize DigitOCR.
        
        Args:
            use_gpu: Use GPU for OCR (requires CUDA)
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.use_gpu = use_gpu
        self._reader = None
        
        if cv2 is None:
            raise ImportError("OpenCV required. Install: pip install opencv-python")
    
    @property
    def reader(self):
        """Lazy load EasyOCR reader"""
        if self._reader is None:
            if easyocr is None:
                raise ImportError("EasyOCR required. Install: pip install easyocr")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        return self._reader
    
    def recognize(self, 
                  image: np.ndarray,
                  line_positions: Optional[List[int]] = None) -> List[DetectedDigit]:
        """
        Recognize all fret numbers in a TAB image.
        
        Args:
            image: TAB region image (BGR or grayscale)
            line_positions: Optional list of line y-positions for filtering
            
        Returns:
            List of DetectedDigit objects
        """
        # Preprocess image
        processed = self._preprocess(image)
        
        # Run OCR
        results = self.reader.readtext(
            processed,
            allowlist='0123456789()',
            paragraph=False,
            detail=1,
            min_size=5,
            text_threshold=0.5,
            low_text=0.3
        )
        
        # Process results
        digits = []
        for result in results:
            bbox, text, confidence = result
            
            if confidence < self.min_confidence:
                continue
            
            # Parse text to get fret number
            fret_value = self._parse_fret_number(text)
            
            if fret_value is None:
                continue
            
            # Calculate center position
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            width = bbox[2][0] - bbox[0][0]
            height = bbox[2][1] - bbox[0][1]
            
            digits.append(DetectedDigit(
                value=fret_value,
                x=x_center,
                y=y_center,
                width=width,
                height=height,
                confidence=confidence,
                raw_text=text
            ))
        
        # Filter by line positions if provided
        if line_positions:
            digits = self._filter_by_lines(digits, line_positions)
        
        # Merge adjacent digits (for two-digit numbers)
        digits = self._merge_adjacent_digits(digits)
        
        return digits
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Threshold to create binary image
        _, binary = cv2.threshold(
            denoised, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary
    
    def _parse_fret_number(self, text: str) -> Optional[int]:
        """
        Parse text to extract fret number.
        
        Handles: "0", "12", "(5)", "(12)" etc.
        """
        # Remove parentheses
        cleaned = text.replace('(', '').replace(')', '').strip()
        
        if not cleaned:
            return None
        
        try:
            value = int(cleaned)
            # Valid fret range is 0-24
            if 0 <= value <= 24:
                return value
            return None
        except ValueError:
            return None
    
    def _filter_by_lines(self, 
                         digits: List[DetectedDigit],
                         line_positions: List[int]) -> List[DetectedDigit]:
        """Filter digits to only those near TAB lines"""
        if not line_positions:
            return digits
        
        # Calculate max distance from line
        if len(line_positions) >= 2:
            spacing = (line_positions[-1] - line_positions[0]) / (len(line_positions) - 1)
            max_distance = spacing * 0.6
        else:
            max_distance = 20
        
        filtered = []
        for digit in digits:
            # Find closest line
            min_dist = min(abs(digit.y - pos) for pos in line_positions)
            if min_dist <= max_distance:
                filtered.append(digit)
        
        return filtered
    
    def _merge_adjacent_digits(self, 
                               digits: List[DetectedDigit],
                               x_threshold: float = 15) -> List[DetectedDigit]:
        """
        Merge adjacent single digits into two-digit numbers.
        
        Example: "1" and "2" close together → "12"
        """
        if len(digits) < 2:
            return digits
        
        # Sort by x position
        sorted_digits = sorted(digits, key=lambda d: d.x)
        
        merged = []
        i = 0
        
        while i < len(sorted_digits):
            current = sorted_digits[i]
            
            # Check if this is a single digit that might be part of a two-digit number
            if (current.value < 10 and 
                i + 1 < len(sorted_digits)):
                
                next_digit = sorted_digits[i + 1]
                
                # Check if they're close enough horizontally and at similar y
                x_close = next_digit.x - current.x < x_threshold
                y_close = abs(next_digit.y - current.y) < 10
                
                if x_close and y_close and next_digit.value < 10:
                    # Merge into two-digit number
                    combined_value = current.value * 10 + next_digit.value
                    
                    if combined_value <= 24:  # Valid fret
                        merged_digit = DetectedDigit(
                            value=combined_value,
                            x=(current.x + next_digit.x) / 2,
                            y=(current.y + next_digit.y) / 2,
                            width=next_digit.x - current.x + max(current.width, next_digit.width),
                            height=max(current.height, next_digit.height),
                            confidence=min(current.confidence, next_digit.confidence),
                            raw_text=f"{current.value}{next_digit.value}"
                        )
                        merged.append(merged_digit)
                        i += 2
                        continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def recognize_region(self,
                        image: np.ndarray,
                        region: Tuple[int, int, int, int]) -> List[DetectedDigit]:
        """
        Recognize digits in a specific region.
        
        Args:
            image: Full image
            region: (x, y, width, height) of region
            
        Returns:
            List of DetectedDigit with coordinates relative to full image
        """
        x, y, w, h = region
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        
        # Recognize
        digits = self.recognize(roi)
        
        # Adjust coordinates to full image
        for digit in digits:
            digit.x += x
            digit.y += y
        
        return digits
