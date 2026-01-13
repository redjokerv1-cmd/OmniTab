"""RegionDetector - Detect TAB regions in sheet music images"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TabRegion:
    """Represents a detected TAB region"""
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    page_number: int
    region_number: int
    
    # Line positions (6 lines)
    line_positions: List[int] = field(default_factory=list)
    
    @property
    def line_spacing(self) -> float:
        """Average spacing between lines"""
        if len(self.line_positions) < 2:
            return 0.0
        spacings = [self.line_positions[i+1] - self.line_positions[i] 
                   for i in range(len(self.line_positions) - 1)]
        return sum(spacings) / len(spacings)


class RegionDetector:
    """
    Detect TAB regions in sheet music images.
    
    TAB is identified by:
    - 6 horizontal lines with consistent spacing
    - Different from standard notation (5 lines)
    """
    
    def __init__(self, 
                 min_line_length: int = 200,
                 line_gap_tolerance: int = 10,
                 min_region_height: int = 50):
        """
        Initialize RegionDetector.
        
        Args:
            min_line_length: Minimum length for line detection
            line_gap_tolerance: Tolerance for line continuity
            min_region_height: Minimum height for valid TAB region
        """
        self.min_line_length = min_line_length
        self.line_gap_tolerance = line_gap_tolerance
        self.min_region_height = min_region_height
        
        if cv2 is None:
            raise ImportError("OpenCV required. Install: pip install opencv-python")
    
    def detect(self, image: np.ndarray, page_number: int = 1) -> List[TabRegion]:
        """
        Detect TAB regions in an image.
        
        Args:
            image: Input image (BGR)
            page_number: Page number for reference
            
        Returns:
            List of detected TabRegion objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect horizontal lines
        horizontal_lines = self._detect_horizontal_lines(gray)
        
        # Group lines into 6-line systems (TAB)
        tab_systems = self._find_six_line_systems(horizontal_lines, gray.shape[0])
        
        # Extract regions
        regions = []
        for i, (y_start, y_end, line_positions) in enumerate(tab_systems):
            # Add padding
            padding = 20
            y1 = max(0, y_start - padding)
            y2 = min(gray.shape[0], y_end + padding)
            
            region_image = image[y1:y2, :].copy()
            
            # Adjust line positions relative to region
            adjusted_lines = [pos - y1 for pos in line_positions]
            
            regions.append(TabRegion(
                image=region_image,
                x=0,
                y=y1,
                width=image.shape[1],
                height=y2 - y1,
                page_number=page_number,
                region_number=i + 1,
                line_positions=adjusted_lines
            ))
        
        return regions
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> List[int]:
        """
        Detect horizontal line y-positions.
        
        Returns list of y-coordinates where horizontal lines exist.
        """
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=self.min_line_length,
            maxLineGap=self.line_gap_tolerance
        )
        
        if lines is None:
            return []
        
        # Extract y-coordinates of horizontal lines
        horizontal_y = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is approximately horizontal
            if abs(y2 - y1) < 5:  # Allow small slope
                avg_y = (y1 + y2) // 2
                horizontal_y.append(avg_y)
        
        # Remove duplicates (lines at similar y positions)
        horizontal_y = sorted(set(horizontal_y))
        
        # Merge nearby lines
        merged = []
        if horizontal_y:
            current = horizontal_y[0]
            for y in horizontal_y[1:]:
                if y - current < 5:  # Merge if within 5 pixels
                    current = (current + y) // 2
                else:
                    merged.append(current)
                    current = y
            merged.append(current)
        
        return merged
    
    def _find_six_line_systems(self, 
                               line_positions: List[int],
                               image_height: int) -> List[Tuple[int, int, List[int]]]:
        """
        Find groups of 6 lines that form TAB systems.
        
        Returns list of (y_start, y_end, line_positions) tuples.
        """
        if len(line_positions) < 6:
            return []
        
        systems = []
        used = set()
        
        for i in range(len(line_positions) - 5):
            if i in used:
                continue
            
            # Check if next 5 lines have consistent spacing
            candidate = line_positions[i:i+6]
            
            # Calculate spacings
            spacings = [candidate[j+1] - candidate[j] for j in range(5)]
            avg_spacing = sum(spacings) / 5
            
            # Check if spacings are consistent (within 20% of average)
            if avg_spacing < 10:  # Too close together
                continue
            
            consistent = all(
                abs(s - avg_spacing) < avg_spacing * 0.3 
                for s in spacings
            )
            
            if consistent:
                # This is likely a TAB system
                systems.append((
                    candidate[0],
                    candidate[5],
                    candidate
                ))
                
                # Mark these lines as used
                for j in range(i, i + 6):
                    used.add(j)
        
        return systems
    
    def detect_with_preprocessing(self, 
                                  image: np.ndarray,
                                  page_number: int = 1) -> List[TabRegion]:
        """
        Detect TAB regions with additional preprocessing.
        
        Uses morphological operations to enhance line detection.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Create horizontal kernel to detect lines
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.min_line_length, 1)
        )
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(
            binary, 
            cv2.MORPH_OPEN, 
            horizontal_kernel
        )
        
        # Find line positions from the detected lines image
        line_positions = []
        for y in range(horizontal_lines.shape[0]):
            if np.sum(horizontal_lines[y, :]) > horizontal_lines.shape[1] * 0.3 * 255:
                line_positions.append(y)
        
        # Merge nearby lines
        merged = []
        if line_positions:
            current = line_positions[0]
            count = 1
            for y in line_positions[1:]:
                if y - current < 5:
                    current = (current * count + y) // (count + 1)
                    count += 1
                else:
                    merged.append(current)
                    current = y
                    count = 1
            merged.append(current)
        
        # Find 6-line systems
        tab_systems = self._find_six_line_systems(merged, gray.shape[0])
        
        # Extract regions
        regions = []
        for i, (y_start, y_end, lines) in enumerate(tab_systems):
            padding = 20
            y1 = max(0, y_start - padding)
            y2 = min(gray.shape[0], y_end + padding)
            
            region_image = image[y1:y2, :].copy()
            adjusted_lines = [pos - y1 for pos in lines]
            
            regions.append(TabRegion(
                image=region_image,
                x=0,
                y=y1,
                width=image.shape[1],
                height=y2 - y1,
                page_number=page_number,
                region_number=i + 1,
                line_positions=adjusted_lines
            ))
        
        return regions
