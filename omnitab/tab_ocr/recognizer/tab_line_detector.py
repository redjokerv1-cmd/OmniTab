"""
TAB Line Detector - Detect the 6 horizontal lines and map digits to strings

Critical for correct GP5 generation:
- Detect 6 TAB lines per system
- Map each recognized digit to correct string (1-6)
- Detect measure boundaries (vertical bar lines)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TabLine:
    """A single TAB line (one of 6 strings)"""
    string_num: int  # 1-6 (1=highest pitch string)
    y_position: float  # Y coordinate
    y_tolerance: float  # How far a digit can be to belong to this line


@dataclass
class TabSystemLines:
    """The 6 lines of a TAB system"""
    lines: List[TabLine]
    y_start: float
    y_end: float
    
    def get_string_for_y(self, y: float) -> int:
        """Determine which string (1-6) a Y position belongs to"""
        best_string = 3  # Default to middle
        best_dist = float('inf')
        
        for line in self.lines:
            dist = abs(y - line.y_position)
            if dist < best_dist and dist < line.y_tolerance:
                best_dist = dist
                best_string = line.string_num
        
        return best_string


@dataclass
class MeasureBoundary:
    """A measure boundary (vertical bar line)"""
    x_position: float
    measure_num: int


class TabLineDetector:
    """
    Detect TAB lines and measure boundaries.
    
    Algorithm:
    1. Use horizontal projection to find dense horizontal line areas
    2. Within each area, find exactly 6 evenly-spaced lines
    3. Detect vertical lines for measure boundaries
    """
    
    def __init__(self, 
                 min_line_density: int = 500,
                 expected_lines_per_system: int = 6,
                 line_spacing_tolerance: float = 0.3):
        """
        Args:
            min_line_density: Minimum horizontal pixel density for line detection
            expected_lines_per_system: Number of lines per TAB system (6 for guitar)
            line_spacing_tolerance: Tolerance for line spacing variance (0.3 = 30%)
        """
        self.min_line_density = min_line_density
        self.expected_lines = expected_lines_per_system
        self.line_spacing_tolerance = line_spacing_tolerance
        
        if cv2 is None:
            raise ImportError("OpenCV required")
    
    def detect(self, image: np.ndarray) -> Tuple[List[TabSystemLines], List[MeasureBoundary]]:
        """
        Detect TAB lines and measure boundaries.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (list of TabSystemLines, list of MeasureBoundaries)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines (TAB staff lines)
        systems = self._detect_tab_systems(binary)
        
        # Detect vertical lines (measure boundaries)
        measures = self._detect_measure_boundaries(binary, systems)
        
        return systems, measures
    
    def _detect_tab_systems(self, binary: np.ndarray) -> List[TabSystemLines]:
        """Detect TAB systems by finding groups of 6 horizontal lines"""
        height, width = binary.shape
        
        # Horizontal projection profile
        h_profile = np.sum(binary, axis=1)
        
        # Find regions with high horizontal density (potential line areas)
        # Use morphology to find horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 4, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        # Find Y positions of horizontal lines
        line_rows = []
        for y in range(height):
            if np.sum(h_lines[y, :]) > width * 0.3:  # Line spans at least 30% of width
                line_rows.append(y)
        
        # Group consecutive rows into lines
        lines = []
        if line_rows:
            current_group = [line_rows[0]]
            for y in line_rows[1:]:
                if y - current_group[-1] <= 3:  # Within 3 pixels
                    current_group.append(y)
                else:
                    lines.append(sum(current_group) / len(current_group))
                    current_group = [y]
            lines.append(sum(current_group) / len(current_group))
        
        # Group lines into systems of 6
        systems = []
        if len(lines) >= 6:
            # Calculate average spacing
            spacings = [lines[i+1] - lines[i] for i in range(len(lines)-1)]
            
            # Find groups of 6 lines with similar spacing
            i = 0
            while i <= len(lines) - 6:
                # Check if next 6 lines have consistent spacing
                group_spacings = [lines[i+j+1] - lines[i+j] for j in range(5)]
                avg_spacing = sum(group_spacings) / 5
                
                # Check variance
                variance = sum((s - avg_spacing)**2 for s in group_spacings) / 5
                if variance < (avg_spacing * self.line_spacing_tolerance) ** 2:
                    # Valid system
                    system_lines = []
                    for j in range(6):
                        system_lines.append(TabLine(
                            string_num=j + 1,
                            y_position=lines[i + j],
                            y_tolerance=avg_spacing * 0.4
                        ))
                    
                    systems.append(TabSystemLines(
                        lines=system_lines,
                        y_start=lines[i] - avg_spacing,
                        y_end=lines[i + 5] + avg_spacing
                    ))
                    i += 6  # Move to next potential system
                else:
                    i += 1
        
        return systems
    
    def _detect_measure_boundaries(self, 
                                    binary: np.ndarray,
                                    systems: List[TabSystemLines]) -> List[MeasureBoundary]:
        """Detect vertical bar lines (measure boundaries)"""
        if not systems:
            return []
        
        height, width = binary.shape
        
        # Vertical line detection kernel
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        # Find X positions of vertical lines
        v_profile = np.sum(v_lines, axis=0)
        
        # Find peaks (bar lines)
        threshold = height * 0.1
        bar_positions = []
        
        in_peak = False
        peak_start = 0
        
        for x in range(width):
            if v_profile[x] > threshold and not in_peak:
                in_peak = True
                peak_start = x
            elif v_profile[x] <= threshold and in_peak:
                in_peak = False
                bar_positions.append((peak_start + x) // 2)
        
        # Convert to measure boundaries
        measures = []
        for i, x in enumerate(bar_positions):
            measures.append(MeasureBoundary(x_position=x, measure_num=i + 1))
        
        return measures
    
    def detect_file(self, path: str) -> Tuple[List[TabSystemLines], List[MeasureBoundary]]:
        """Detect from file"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read: {path}")
        return self.detect(image)


# CLI
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tab_line_detector.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Detecting TAB lines: {path}")
    print("=" * 60)
    
    detector = TabLineDetector()
    systems, measures = detector.detect_file(path)
    
    print(f"Found {len(systems)} TAB systems")
    for i, system in enumerate(systems):
        print(f"\nSystem {i+1}: Y={system.y_start:.0f}-{system.y_end:.0f}")
        for line in system.lines:
            print(f"  String {line.string_num}: Y={line.y_position:.0f} (+/-{line.y_tolerance:.0f})")
    
    print(f"\nFound {len(measures)} measure boundaries")
    for m in measures[:10]:
        print(f"  Measure {m.measure_num}: X={m.x_position:.0f}")
    if len(measures) > 10:
        print(f"  ... +{len(measures) - 10} more")
