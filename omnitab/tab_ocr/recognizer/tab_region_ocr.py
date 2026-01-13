"""
TAB Region OCR - Only OCR within detected TAB systems

Creative approach:
1. First detect TAB line systems with horizontal projection
2. Crop each system region
3. Run OCR only on cropped regions
4. This eliminates false positives from staff notation areas

Expected improvement:
- No more unmapped digits from staff areas
- Higher accuracy in TAB regions
- Better 2-digit number handling
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .horizontal_projection import HorizontalProjection, TabStaffSystem
from .simple_binary_ocr import SimpleBinaryOCR


@dataclass
class RegionDigit:
    """Digit recognized within a TAB region"""
    value: int
    x: float  # Absolute X in original image
    y: float  # Absolute Y in original image
    string: int  # 1-6, mapped directly
    confidence: float
    system_idx: int


class TabRegionOCR:
    """
    OCR that only processes TAB system regions.
    
    Algorithm:
    1. Detect TAB systems with horizontal projection
    2. For each system:
       a. Crop the region (with padding)
       b. Remove staff lines
       c. Run OCR
       d. Map digits to strings immediately
    3. Return only valid TAB digits
    """
    
    def __init__(self,
                 padding: int = 10,
                 use_gpu: bool = False,
                 min_confidence: float = 0.3):
        self.padding = padding
        self.line_detector = HorizontalProjection()
        self.ocr = SimpleBinaryOCR(
            scale_factor=3,
            min_confidence=min_confidence,
            use_gpu=use_gpu
        )
    
    def process(self, image: np.ndarray) -> Dict:
        """
        Process image and return only TAB region digits.
        
        Returns:
            dict with 'digits', 'systems', 'stats'
        """
        height, width = image.shape[:2]
        
        # Step 1: Detect TAB systems
        systems = self.line_detector.detect(image)
        
        if not systems:
            return {
                'digits': [],
                'systems': [],
                'stats': {'total_digits': 0, 'systems': 0}
            }
        
        all_digits = []
        
        # Step 2: Process each TAB system
        for sys_idx, system in enumerate(systems):
            # Crop region with padding
            y_start = max(0, int(system.y_start) - self.padding)
            y_end = min(height, int(system.y_end) + self.padding)
            
            region = image[y_start:y_end, :]
            
            # Remove staff lines from region
            region_clean = self._remove_lines(region)
            
            # Run OCR on clean region
            region_digits = self._ocr_region(region_clean)
            
            # Map to strings and convert to absolute coordinates
            for digit in region_digits:
                # Adjust Y to absolute
                abs_y = digit['y'] + y_start
                
                # Map to string
                string = system.get_string_for_y(abs_y)
                
                if string > 0:
                    all_digits.append(RegionDigit(
                        value=digit['value'],
                        x=digit['x'],
                        y=abs_y,
                        string=string,
                        confidence=digit['confidence'],
                        system_idx=sys_idx
                    ))
        
        # Deduplicate
        deduped = self._deduplicate(all_digits)
        
        return {
            'digits': deduped,
            'systems': systems,
            'stats': {
                'total_digits': len(deduped),
                'systems': len(systems),
                'per_system': [sum(1 for d in deduped if d.system_idx == i) for i in range(len(systems))]
            }
        }
    
    def _remove_lines(self, region: np.ndarray) -> np.ndarray:
        """Remove horizontal staff lines from region"""
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        # Remove lines
        result = gray.copy()
        result[lines == 255] = 255
        
        return result
    
    def _ocr_region(self, region: np.ndarray) -> List[Dict]:
        """Run OCR on a clean region"""
        # Binary threshold
        _, binary = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digits = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter by size
            if w < 3 or h < 5 or w > 50 or h > 40:
                continue
            
            # Extract digit region
            digit_img = region[y:y+h, x:x+w]
            
            # Scale up small digits
            if w < 15 or h < 15:
                scale = 3
                digit_img = cv2.resize(digit_img, (w*scale, h*scale), 
                                       interpolation=cv2.INTER_CUBIC)
            
            # OCR
            try:
                results = self.ocr.reader.readtext(
                    digit_img,
                    allowlist='0123456789',
                    detail=1,
                    paragraph=False
                )
                
                if results:
                    text = results[0][1].strip()
                    conf = results[0][2]
                    
                    if conf >= 0.2 and text:
                        # Try to parse as integer
                        try:
                            value = int(text)
                            if 0 <= value <= 24:  # Valid fret range
                                digits.append({
                                    'value': value,
                                    'x': x + w/2,
                                    'y': y + h/2,
                                    'confidence': conf
                                })
                        except ValueError:
                            # Single digit fallback
                            for char in text:
                                if char.isdigit():
                                    digits.append({
                                        'value': int(char),
                                        'x': x + w/2,
                                        'y': y + h/2,
                                        'confidence': conf * 0.8
                                    })
                                    break
                        
            except Exception:
                continue
        
        return digits
    
    def _deduplicate(self, 
                     digits: List[RegionDigit],
                     x_threshold: float = 12) -> List[RegionDigit]:
        """Remove duplicate digits"""
        if not digits:
            return digits
        
        # Sort by system, string, x
        sorted_digits = sorted(digits, key=lambda d: (d.system_idx, d.string, d.x))
        
        result = [sorted_digits[0]]
        
        for digit in sorted_digits[1:]:
            prev = result[-1]
            
            # Same system, same string, close X
            if (digit.system_idx == prev.system_idx and
                digit.string == prev.string and
                abs(digit.x - prev.x) < x_threshold):
                # Keep higher value (likely 2-digit) or higher confidence
                if digit.value > prev.value or digit.confidence > prev.confidence:
                    result[-1] = digit
            else:
                result.append(digit)
        
        return result


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tab_region_ocr.py <image_path>")
        sys.exit(1)
    
    image = cv2.imread(sys.argv[1])
    
    ocr = TabRegionOCR()
    result = ocr.process(image)
    
    print(f"Systems: {result['stats']['systems']}")
    print(f"Total digits: {result['stats']['total_digits']}")
    print(f"Per system: {result['stats']['per_system']}")
    
    print("\nFirst 20 digits:")
    for d in sorted(result['digits'], key=lambda x: (x.system_idx, x.x))[:20]:
        print(f"  S{d.string} F{d.value:2d} at ({d.x:.0f}, {d.y:.0f}) sys={d.system_idx}")
