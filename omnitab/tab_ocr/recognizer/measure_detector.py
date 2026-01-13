"""
Measure Detector - Detect vertical bar lines (measure boundaries)

Uses vertical projection profile to find bar lines that span
all 6 TAB lines within a system.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .horizontal_projection import TabStaffSystem


@dataclass
class Measure:
    """A measure with start and end X positions"""
    x_start: float
    x_end: float
    measure_num: int
    
    @property
    def width(self) -> float:
        return self.x_end - self.x_start


class MeasureDetector:
    """
    Detect measure boundaries using vertical projection.
    
    Algorithm:
    1. For each TAB system, extract the region
    2. Apply vertical projection (sum columns)
    3. Find peaks that span the full system height
    4. These are the bar lines (measure boundaries)
    """
    
    def __init__(self,
                 min_bar_height_ratio: float = 0.7,
                 min_measure_width: int = 30):
        """
        Args:
            min_bar_height_ratio: Bar must span at least this ratio of system height
            min_measure_width: Minimum width of a measure in pixels
        """
        self.min_bar_height_ratio = min_bar_height_ratio
        self.min_measure_width = min_measure_width
        
        if cv2 is None:
            raise ImportError("OpenCV required")
    
    def detect(self, 
               image: np.ndarray,
               systems: List[TabStaffSystem]) -> List[List[Measure]]:
        """
        Detect measures in each TAB system.
        
        Args:
            image: Input image
            systems: List of detected TAB systems
            
        Returns:
            List of measures for each system
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        all_measures = []
        
        for system in systems:
            measures = self._detect_in_system(binary, system)
            all_measures.append(measures)
        
        return all_measures
    
    def _detect_in_system(self, 
                          binary: np.ndarray,
                          system: TabStaffSystem) -> List[Measure]:
        """Detect measures in a single TAB system"""
        height, width = binary.shape
        
        # Extract system region
        y_start = max(0, int(system.y_start))
        y_end = min(height, int(system.y_end))
        system_region = binary[y_start:y_end, :]
        
        system_height = y_end - y_start
        
        # Vertical projection
        v_profile = np.sum(system_region, axis=0) / 255
        
        # Find columns with strong vertical content (bar lines)
        # A bar line should span most of the system height
        min_bar_pixels = system_height * self.min_bar_height_ratio
        
        bar_positions = []
        in_bar = False
        bar_start = 0
        
        for x in range(width):
            if v_profile[x] >= min_bar_pixels and not in_bar:
                in_bar = True
                bar_start = x
            elif v_profile[x] < min_bar_pixels and in_bar:
                in_bar = False
                bar_center = (bar_start + x) // 2
                bar_positions.append(bar_center)
        
        # Add image boundaries if no bars found
        if not bar_positions:
            bar_positions = [0, width - 1]
        else:
            # Ensure we have start and end
            if bar_positions[0] > 50:
                bar_positions.insert(0, 0)
            if bar_positions[-1] < width - 50:
                bar_positions.append(width - 1)
        
        # Filter out bars that are too close
        filtered_bars = [bar_positions[0]]
        for bar in bar_positions[1:]:
            if bar - filtered_bars[-1] >= self.min_measure_width:
                filtered_bars.append(bar)
        
        # Create measures
        measures = []
        for i in range(len(filtered_bars) - 1):
            measures.append(Measure(
                x_start=filtered_bars[i],
                x_end=filtered_bars[i + 1],
                measure_num=i + 1
            ))
        
        return measures


# CLI
if __name__ == '__main__':
    import sys
    from omnitab.tab_ocr.recognizer.horizontal_projection import HorizontalProjection
    
    if len(sys.argv) < 2:
        print("Usage: python measure_detector.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Detecting measures: {path}")
    print("=" * 60)
    
    image = cv2.imread(path)
    
    # First detect TAB lines
    line_detector = HorizontalProjection()
    systems = line_detector.detect(image)
    print(f"Found {len(systems)} TAB systems")
    
    # Then detect measures
    measure_detector = MeasureDetector()
    all_measures = measure_detector.detect(image, systems)
    
    for i, measures in enumerate(all_measures):
        print(f"\nSystem {i+1}: {len(measures)} measures")
        for m in measures[:5]:
            print(f"  Measure {m.measure_num}: X={m.x_start:.0f}-{m.x_end:.0f} ({m.width:.0f}px)")
        if len(measures) > 5:
            print(f"  ... +{len(measures) - 5} more")
