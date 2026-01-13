"""MeasureDetector - Detect measure bars in TAB images"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from ..models import TabMeasure, TabChord


@dataclass
class DetectedBar:
    """A detected measure bar line"""
    x: int
    y_start: int
    y_end: int
    bar_type: str  # 'single', 'double', 'repeat_start', 'repeat_end', 'final'
    confidence: float


class MeasureDetector:
    """
    Detect measure bar lines in TAB images.
    
    Identifies:
    - Single bar lines
    - Double bar lines
    - Repeat signs
    - Final bar lines
    """
    
    def __init__(self, 
                 min_bar_height: int = 30,
                 bar_thickness_range: Tuple[int, int] = (1, 5)):
        """
        Initialize MeasureDetector.
        
        Args:
            min_bar_height: Minimum height for bar line
            bar_thickness_range: (min, max) thickness of bar lines
        """
        self.min_bar_height = min_bar_height
        self.bar_thickness_range = bar_thickness_range
        
        if cv2 is None:
            raise ImportError("OpenCV required. Install: pip install opencv-python")
    
    def detect_bars(self, 
                    image: np.ndarray,
                    line_positions: Optional[List[int]] = None) -> List[DetectedBar]:
        """
        Detect bar lines in a TAB image.
        
        Args:
            image: TAB region image
            line_positions: Optional y-positions of TAB lines
            
        Returns:
            List of DetectedBar objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Create vertical kernel to detect bar lines
        kernel_height = self.min_bar_height
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.bar_thickness_range[1], kernel_height)
        )
        
        # Detect vertical lines
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if h < self.min_bar_height:
                continue
            
            if not (self.bar_thickness_range[0] <= w <= self.bar_thickness_range[1] * 2):
                continue
            
            # Check if line spans across TAB area
            if line_positions and len(line_positions) >= 2:
                expected_y_start = line_positions[0] - 5
                expected_y_end = line_positions[-1] + 5
                
                if not (y <= expected_y_start + 10 and y + h >= expected_y_end - 10):
                    continue
            
            # Determine bar type
            bar_type = self._classify_bar(gray, x, y, w, h)
            
            bars.append(DetectedBar(
                x=x + w // 2,
                y_start=y,
                y_end=y + h,
                bar_type=bar_type,
                confidence=0.8
            ))
        
        # Sort by x position
        bars.sort(key=lambda b: b.x)
        
        # Merge nearby bars (double bars, repeats)
        merged = self._merge_nearby_bars(bars)
        
        return merged
    
    def _classify_bar(self, 
                      gray: np.ndarray,
                      x: int, y: int, w: int, h: int) -> str:
        """Classify the type of bar line"""
        # Check for thick bar (final)
        if w > self.bar_thickness_range[1]:
            return 'final'
        
        # Check for repeat dots nearby
        # Look for dots on left (repeat end) or right (repeat start)
        region_left = gray[y:y+h, max(0, x-20):x]
        region_right = gray[y:y+h, x+w:min(gray.shape[1], x+w+20)]
        
        left_has_dots = self._has_dots(region_left)
        right_has_dots = self._has_dots(region_right)
        
        if right_has_dots:
            return 'repeat_start'
        if left_has_dots:
            return 'repeat_end'
        
        return 'single'
    
    def _has_dots(self, region: np.ndarray) -> bool:
        """Check if a region contains repeat dots"""
        if region.size == 0:
            return False
        
        _, binary = cv2.threshold(region, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find small circular contours (dots)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        dot_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Small circular shape
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                if 0.5 < aspect < 2:  # Roughly square
                    dot_count += 1
        
        return dot_count >= 2  # Need at least 2 dots for repeat
    
    def _merge_nearby_bars(self, 
                          bars: List[DetectedBar],
                          x_threshold: int = 15) -> List[DetectedBar]:
        """Merge nearby bar lines (for double bars)"""
        if len(bars) < 2:
            return bars
        
        merged = []
        i = 0
        
        while i < len(bars):
            current = bars[i]
            
            # Check if next bar is close
            if i + 1 < len(bars) and bars[i + 1].x - current.x < x_threshold:
                next_bar = bars[i + 1]
                
                # Merge into double bar
                merged.append(DetectedBar(
                    x=(current.x + next_bar.x) // 2,
                    y_start=min(current.y_start, next_bar.y_start),
                    y_end=max(current.y_end, next_bar.y_end),
                    bar_type='double',
                    confidence=min(current.confidence, next_bar.confidence)
                ))
                i += 2
            else:
                merged.append(current)
                i += 1
        
        return merged
    
    def split_into_measures(self,
                           chords: List[TabChord],
                           bars: List[DetectedBar],
                           time_signature: Tuple[int, int] = (4, 4)) -> List[TabMeasure]:
        """
        Split chords into measures using detected bar lines.
        
        Args:
            chords: List of TabChord objects
            bars: List of detected bar lines
            time_signature: Time signature (beats, beat_value)
            
        Returns:
            List of TabMeasure objects
        """
        if not chords:
            return []
        
        if not bars:
            # No bars detected - put all in one measure
            return [TabMeasure(
                number=1,
                chords=chords,
                time_signature=time_signature
            )]
        
        measures = []
        current_chords = []
        measure_number = 1
        bar_idx = 0
        
        # Add virtual bar at start if first bar is not at the beginning
        if bars and bars[0].x > chords[0].x_position + 20:
            bars = [DetectedBar(
                x=0, y_start=0, y_end=0, 
                bar_type='virtual', confidence=1.0
            )] + bars
        
        for chord in chords:
            # Check if we've passed a bar line
            while (bar_idx < len(bars) and 
                   chord.x_position > bars[bar_idx].x):
                # Save current measure
                if current_chords:
                    bar = bars[bar_idx] if bar_idx < len(bars) else None
                    
                    measure = TabMeasure(
                        number=measure_number,
                        chords=current_chords.copy(),
                        time_signature=time_signature,
                        has_repeat_start=bar and bar.bar_type == 'repeat_start',
                        has_repeat_end=bar and bar.bar_type == 'repeat_end'
                    )
                    measures.append(measure)
                    
                    current_chords = []
                    measure_number += 1
                
                bar_idx += 1
            
            current_chords.append(chord)
        
        # Don't forget last measure
        if current_chords:
            measures.append(TabMeasure(
                number=measure_number,
                chords=current_chords,
                time_signature=time_signature
            ))
        
        return measures
