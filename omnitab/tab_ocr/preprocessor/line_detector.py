"""LineDetector - Detect and refine the 6 TAB lines within a region"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TabLines:
    """Detected TAB lines with their properties"""
    positions: List[int]      # Y positions of the 6 lines
    spacing: float            # Average spacing between lines
    thickness: float          # Average line thickness
    confidence: float         # Detection confidence (0-1)
    
    @property
    def is_valid(self) -> bool:
        """Check if we have exactly 6 lines"""
        return len(self.positions) == 6
    
    def get_string_for_y(self, y: float) -> int:
        """
        Get the guitar string number (1-6) for a given y position.
        
        String 1 is the topmost line (highest pitch on guitar).
        """
        if not self.is_valid:
            return 1
        
        # Find closest line
        distances = [abs(y - pos) for pos in self.positions]
        closest_idx = distances.index(min(distances))
        
        return closest_idx + 1
    
    def get_y_range_for_string(self, string: int) -> Tuple[float, float]:
        """
        Get the y-range that belongs to a specific string.
        
        Args:
            string: String number (1-6)
            
        Returns:
            (y_min, y_max) tuple
        """
        if not self.is_valid or not 1 <= string <= 6:
            return (0, 0)
        
        idx = string - 1
        half_spacing = self.spacing / 2
        
        y_center = self.positions[idx]
        return (y_center - half_spacing, y_center + half_spacing)


class LineDetector:
    """
    Detect and refine the positions of 6 TAB lines.
    
    Works on a pre-extracted TAB region to precisely locate each line.
    """
    
    def __init__(self, 
                 expected_lines: int = 6,
                 line_search_range: int = 10):
        """
        Initialize LineDetector.
        
        Args:
            expected_lines: Number of lines to detect (6 for guitar TAB)
            line_search_range: Range to search for line refinement
        """
        self.expected_lines = expected_lines
        self.line_search_range = line_search_range
        
        if cv2 is None:
            raise ImportError("OpenCV required. Install: pip install opencv-python")
    
    def detect(self, 
               region_image: np.ndarray,
               initial_positions: Optional[List[int]] = None) -> TabLines:
        """
        Detect TAB lines in a region.
        
        Args:
            region_image: Image of the TAB region
            initial_positions: Optional initial line positions to refine
            
        Returns:
            TabLines object with detected line positions
        """
        # Convert to grayscale
        if len(region_image.shape) == 3:
            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = region_image.copy()
        
        if initial_positions and len(initial_positions) == 6:
            # Refine initial positions
            positions = self._refine_positions(gray, initial_positions)
        else:
            # Detect from scratch
            positions = self._detect_lines(gray)
        
        # Calculate metrics
        if len(positions) >= 2:
            spacings = [positions[i+1] - positions[i] 
                       for i in range(len(positions) - 1)]
            spacing = sum(spacings) / len(spacings)
        else:
            spacing = 0.0
        
        # Estimate line thickness
        thickness = self._estimate_thickness(gray, positions)
        
        # Calculate confidence
        confidence = self._calculate_confidence(positions, gray.shape[0])
        
        return TabLines(
            positions=positions,
            spacing=spacing,
            thickness=thickness,
            confidence=confidence
        )
    
    def _detect_lines(self, gray: np.ndarray) -> List[int]:
        """Detect line positions from scratch"""
        # Calculate horizontal projection
        projection = np.sum(255 - gray, axis=1)
        
        # Find peaks in projection (lines are dark = high projection value)
        peaks = self._find_peaks(projection)
        
        # If we have too many peaks, filter to find the 6 most consistent
        if len(peaks) > self.expected_lines:
            peaks = self._filter_to_best_group(peaks)
        
        return sorted(peaks)[:self.expected_lines]
    
    def _refine_positions(self, 
                          gray: np.ndarray, 
                          initial: List[int]) -> List[int]:
        """Refine initial line positions using local search"""
        refined = []
        
        for pos in initial:
            # Search in range around initial position
            start = max(0, pos - self.line_search_range)
            end = min(gray.shape[0], pos + self.line_search_range + 1)
            
            # Find the darkest row (line) in this range
            best_pos = pos
            best_darkness = 0
            
            for y in range(start, end):
                darkness = np.sum(255 - gray[y, :])
                if darkness > best_darkness:
                    best_darkness = darkness
                    best_pos = y
            
            refined.append(best_pos)
        
        return refined
    
    def _find_peaks(self, projection: np.ndarray) -> List[int]:
        """Find peaks in horizontal projection"""
        # Smooth the projection
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Find local maxima
        peaks = []
        threshold = np.mean(smoothed) + np.std(smoothed)
        
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > smoothed[i-1] and 
                smoothed[i] > smoothed[i+1] and
                smoothed[i] > threshold):
                peaks.append(i)
        
        return peaks
    
    def _filter_to_best_group(self, peaks: List[int]) -> List[int]:
        """
        Filter peaks to find the best group of 6 with consistent spacing.
        """
        if len(peaks) <= self.expected_lines:
            return peaks
        
        best_group = peaks[:self.expected_lines]
        best_variance = float('inf')
        
        # Try all combinations of 6 consecutive peaks
        for i in range(len(peaks) - self.expected_lines + 1):
            group = peaks[i:i + self.expected_lines]
            
            # Calculate spacing variance
            spacings = [group[j+1] - group[j] for j in range(len(group) - 1)]
            variance = np.var(spacings)
            
            if variance < best_variance:
                best_variance = variance
                best_group = group
        
        return best_group
    
    def _estimate_thickness(self, 
                            gray: np.ndarray, 
                            positions: List[int]) -> float:
        """Estimate average line thickness"""
        if not positions:
            return 1.0
        
        thicknesses = []
        
        for pos in positions:
            # Count consecutive dark pixels around line position
            thickness = 1
            threshold = 200  # Gray value threshold
            
            # Check above
            for y in range(pos - 1, max(0, pos - 10), -1):
                if np.mean(gray[y, :]) < threshold:
                    thickness += 1
                else:
                    break
            
            # Check below
            for y in range(pos + 1, min(gray.shape[0], pos + 10)):
                if np.mean(gray[y, :]) < threshold:
                    thickness += 1
                else:
                    break
            
            thicknesses.append(thickness)
        
        return sum(thicknesses) / len(thicknesses) if thicknesses else 1.0
    
    def _calculate_confidence(self, 
                             positions: List[int],
                             image_height: int) -> float:
        """Calculate detection confidence"""
        if len(positions) != self.expected_lines:
            return 0.0
        
        # Check spacing consistency
        spacings = [positions[i+1] - positions[i] 
                   for i in range(len(positions) - 1)]
        
        if not spacings:
            return 0.0
        
        avg_spacing = sum(spacings) / len(spacings)
        
        if avg_spacing < 10:  # Too close
            return 0.3
        
        # Calculate spacing variance
        variance = np.var(spacings)
        max_variance = avg_spacing ** 2
        
        spacing_score = 1.0 - min(variance / max_variance, 1.0)
        
        # Check if lines span reasonable portion of region
        span = positions[-1] - positions[0]
        expected_span = image_height * 0.4  # Expect TAB to be ~40% of region
        span_score = 1.0 - abs(span - expected_span) / expected_span
        span_score = max(0.0, min(1.0, span_score))
        
        return 0.7 * spacing_score + 0.3 * span_score
