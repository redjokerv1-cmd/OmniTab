"""
Smart OCR - Creative approach for better digit recognition

Key insight: Contour-based approach fails for 2-digit numbers because
TAB lines split them into separate contours.

New approach:
1. Use horizontal projection to find TAB systems
2. For each line in a system, scan horizontally with a sliding window
3. OCR each window and merge overlapping results
4. Prefer 2-digit numbers over single digits in overlapping regions

This is a creative, custom solution tailored for TAB notation.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import easyocr

from .horizontal_projection import HorizontalProjection, TabStaffSystem


@dataclass 
class SmartDigit:
    """Recognized digit with smart merging"""
    value: int
    x: float
    y: float
    string: int
    confidence: float
    width: float  # For overlap detection


class SmartTabOCR:
    """
    Smart OCR that scans TAB lines with sliding windows.
    
    Creative approach:
    1. Remove staff lines
    2. For each string position, scan horizontally
    3. Use overlapping windows to catch 2-digit numbers
    4. Merge overlapping detections, preferring 2-digit
    """
    
    def __init__(self,
                 window_width: int = 25,
                 window_stride: int = 8,
                 scale_factor: int = 4,
                 min_confidence: float = 0.3):
        self.window_width = window_width
        self.window_stride = window_stride
        self.scale_factor = scale_factor
        self.min_confidence = min_confidence
        
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.line_detector = HorizontalProjection()
    
    def process(self, image: np.ndarray) -> Dict:
        """Process image with smart scanning"""
        height, width = image.shape[:2]
        
        # Detect TAB systems
        systems = self.line_detector.detect(image)
        
        if not systems:
            return {'digits': [], 'systems': [], 'stats': {}}
        
        # Prepare image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Remove staff lines
        clean = self._remove_lines(gray)
        
        all_digits = []
        
        for sys_idx, system in enumerate(systems):
            # Process each string line
            for string_num, line_y in enumerate(system.line_y_positions, 1):
                digits = self._scan_line(clean, line_y, width, string_num, sys_idx)
                all_digits.extend(digits)
        
        # Merge overlapping detections
        merged = self._merge_overlapping(all_digits)
        
        return {
            'digits': merged,
            'systems': systems,
            'stats': {
                'total_digits': len(merged),
                'systems': len(systems),
                'raw_detections': len(all_digits)
            }
        }
    
    def _remove_lines(self, gray: np.ndarray) -> np.ndarray:
        """Remove horizontal staff lines"""
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        clean = gray.copy()
        clean[lines == 255] = 255
        
        return clean
    
    def _scan_line(self,
                   image: np.ndarray,
                   line_y: float,
                   width: int,
                   string_num: int,
                   sys_idx: int) -> List[SmartDigit]:
        """Scan a single string line with sliding windows"""
        digits = []
        
        # Window height centered on line
        h = 20
        y_start = max(0, int(line_y) - h)
        y_end = min(image.shape[0], int(line_y) + h)
        
        # Sliding window
        for x in range(0, width - self.window_width, self.window_stride):
            window = image[y_start:y_end, x:x+self.window_width]
            
            # Skip mostly white windows
            if np.mean(window) > 240:
                continue
            
            # Scale up
            scaled = cv2.resize(window, 
                               (self.window_width * self.scale_factor, 
                                (y_end - y_start) * self.scale_factor),
                               interpolation=cv2.INTER_CUBIC)
            
            # OCR
            results = self.reader.readtext(
                scaled,
                allowlist='0123456789',
                detail=1,
                paragraph=False
            )
            
            for result in results:
                text = result[1].strip()
                conf = result[2]
                
                if conf >= self.min_confidence and text:
                    try:
                        value = int(text)
                        # High frets (19+) require higher confidence
                        if value >= 19 and conf < 0.7:
                            continue
                        if 0 <= value <= 24:
                            # Calculate center X
                            bbox = result[0]
                            bbox_center_x = (bbox[0][0] + bbox[2][0]) / 2 / self.scale_factor
                            digit_x = x + bbox_center_x
                            digit_width = (bbox[2][0] - bbox[0][0]) / self.scale_factor
                            
                            digits.append(SmartDigit(
                                value=value,
                                x=digit_x,
                                y=line_y,
                                string=string_num,
                                confidence=conf,
                                width=max(digit_width, 5)
                            ))
                    except ValueError:
                        continue
        
        return digits
    
    def _merge_overlapping(self, 
                           digits: List[SmartDigit],
                           overlap_threshold: float = 10) -> List[SmartDigit]:
        """Merge overlapping detections, preferring 2-digit numbers"""
        if not digits:
            return []
        
        # Sort by string, then X
        sorted_digits = sorted(digits, key=lambda d: (d.string, d.x))
        
        merged = []
        i = 0
        
        while i < len(sorted_digits):
            current = sorted_digits[i]
            
            # Collect overlapping digits
            group = [current]
            j = i + 1
            
            while j < len(sorted_digits):
                next_d = sorted_digits[j]
                
                # Same string and overlapping X
                if (next_d.string == current.string and 
                    abs(next_d.x - current.x) < overlap_threshold):
                    group.append(next_d)
                    j += 1
                else:
                    break
            
            # Pick best from group
            # Prefer: 2-digit > 1-digit, high confidence
            best = max(group, key=lambda d: (d.value >= 10, d.confidence))
            merged.append(best)
            
            i = j
        
        return merged


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python smart_ocr.py <image_path>")
        sys.exit(1)
    
    image = cv2.imread(sys.argv[1])
    
    ocr = SmartTabOCR()
    result = ocr.process(image)
    
    print(f"Systems: {result['stats']['systems']}")
    print(f"Raw detections: {result['stats']['raw_detections']}")
    print(f"After merge: {result['stats']['total_digits']}")
    
    # Count 2-digit numbers
    two_digit = sum(1 for d in result['digits'] if d.value >= 10)
    print(f"2-digit numbers: {two_digit}")
    
    print("\nFirst 25 digits:")
    for d in sorted(result['digits'], key=lambda x: (x.string, x.x))[:25]:
        print(f"  S{d.string} F{d.value:2d} at X={d.x:.0f} (conf={d.confidence:.2f})")
