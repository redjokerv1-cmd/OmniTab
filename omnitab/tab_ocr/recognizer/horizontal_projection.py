"""
Horizontal Projection - Accurate TAB 6-Line Detection

This is the correct approach for TAB staff line detection.
Instead of finding "line-like things" (Hough), we sum pixel values
per row to find the 6 equally-spaced peaks.

Algorithm:
1. Convert to binary (black = 1, white = 0)
2. Sum each row (horizontal projection profile)
3. Find peaks in the profile
4. Look for groups of 6 equally-spaced peaks
5. Return the Y coordinates of the 6 lines

This is robust because:
- TAB lines span most of the image width → strong peaks
- 6 lines are equally spaced → easy to identify
- Noise/other lines don't have this pattern
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TabStaffSystem:
    """A TAB staff system with 6 lines"""
    line_y_positions: List[float]  # Y coordinates of 6 lines
    line_spacing: float  # Average spacing between lines
    y_start: float  # Top of system
    y_end: float  # Bottom of system
    confidence: float  # Detection confidence
    
    def get_string_for_y(self, y: float) -> int:
        """
        Map a Y position to string number (1-6).
        
        String 1 = highest line (top)
        String 6 = lowest line (bottom)
        """
        if not self.line_y_positions:
            return 3  # Default to middle
        
        # Find closest line
        min_dist = float('inf')
        best_string = 1
        
        for i, line_y in enumerate(self.line_y_positions):
            dist = abs(y - line_y)
            if dist < min_dist:
                min_dist = dist
                best_string = i + 1
        
        # Check if within tolerance (half line spacing)
        if min_dist > self.line_spacing * 0.6:
            return 0  # Not on any line
        
        return best_string


class HorizontalProjection:
    """
    Detect TAB staff lines using horizontal projection profile.
    
    This is the correct OMR technique for staff line detection.
    """
    
    def __init__(self,
                 min_line_width_ratio: float = 0.3,
                 peak_threshold_ratio: float = 0.5,
                 spacing_tolerance: float = 0.2,
                 min_system_gap: int = 50):
        """
        Args:
            min_line_width_ratio: Minimum line width as ratio of image width
            peak_threshold_ratio: Peak detection threshold as ratio of max peak
            spacing_tolerance: Tolerance for equal spacing check (0.2 = 20%)
            min_system_gap: Minimum Y gap between TAB systems
        """
        self.min_line_width_ratio = min_line_width_ratio
        self.peak_threshold_ratio = peak_threshold_ratio
        self.spacing_tolerance = spacing_tolerance
        self.min_system_gap = min_system_gap
        
        if cv2 is None:
            raise ImportError("OpenCV required: pip install opencv-python")
    
    def detect(self, image: np.ndarray) -> List[TabStaffSystem]:
        """
        Detect all TAB staff systems in the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of TabStaffSystem objects
        """
        # Step 1: Convert to grayscale and binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        height, width = binary.shape
        
        # Step 2: Compute horizontal projection profile
        # Sum of black pixels (value 255 in binary_inv) per row
        h_profile = np.sum(binary, axis=1) / 255  # Normalize to pixel count
        
        # Step 3: Find peaks (rows with many black pixels = lines)
        min_peak_value = width * self.min_line_width_ratio
        peaks = self._find_peaks(h_profile, min_peak_value)
        
        print(f"Found {len(peaks)} potential line peaks")
        
        # Step 4: Group peaks into systems of 6
        systems = self._group_into_systems(peaks, h_profile)
        
        return systems
    
    def _find_peaks(self, profile: np.ndarray, threshold: float) -> List[int]:
        """Find peak positions in the projection profile"""
        peaks = []
        
        # Simple peak detection: local maxima above threshold
        for i in range(1, len(profile) - 1):
            if profile[i] >= threshold:
                # Check if local maximum
                if profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
                    # Avoid duplicates (merge nearby peaks)
                    if not peaks or i - peaks[-1] > 3:
                        peaks.append(i)
        
        return peaks
    
    def _group_into_systems(self, 
                            peaks: List[int],
                            profile: np.ndarray) -> List[TabStaffSystem]:
        """Group peaks into systems of 6 equally-spaced lines"""
        systems = []
        used_peaks = set()
        
        # First pass: Try exact 6-peak systems
        for start_idx in range(len(peaks)):
            if peaks[start_idx] in used_peaks:
                continue
            
            system = self._try_build_system(peaks, start_idx, used_peaks)
            
            if system:
                systems.append(system)
                for y in system.line_y_positions:
                    used_peaks.add(int(y))
        
        # Second pass: Try systems with interpolation (for missing lines)
        for start_idx in range(len(peaks)):
            if peaks[start_idx] in used_peaks:
                continue
            
            system = self._try_build_system_with_interpolation(peaks, start_idx, used_peaks)
            
            if system:
                systems.append(system)
                for y in system.line_y_positions:
                    used_peaks.add(int(y))
        
        return systems
    
    def _try_build_system(self,
                          peaks: List[int],
                          start_idx: int,
                          used_peaks: set) -> Optional[TabStaffSystem]:
        """Try to build a 6-line system starting from given peak"""
        if start_idx + 5 >= len(peaks):
            return None
        
        # Get 6 consecutive peaks
        candidate_peaks = peaks[start_idx:start_idx + 6]
        
        # Check if any are already used
        if any(p in used_peaks for p in candidate_peaks):
            return None
        
        # Calculate spacings
        spacings = [candidate_peaks[i+1] - candidate_peaks[i] for i in range(5)]
        avg_spacing = sum(spacings) / 5
        
        if avg_spacing < 5:  # Too close together
            return None
        
        # Check if spacings are equal (within tolerance)
        for spacing in spacings:
            if abs(spacing - avg_spacing) / avg_spacing > self.spacing_tolerance:
                return None
        
        # Valid system found!
        confidence = 1.0 - (max(abs(s - avg_spacing) for s in spacings) / avg_spacing)
        
        return TabStaffSystem(
            line_y_positions=[float(p) for p in candidate_peaks],
            line_spacing=avg_spacing,
            y_start=candidate_peaks[0] - avg_spacing / 2,
            y_end=candidate_peaks[5] + avg_spacing / 2,
            confidence=confidence
        )
    
    def _try_build_system_with_interpolation(self,
                                              peaks: List[int],
                                              start_idx: int,
                                              used_peaks: set) -> Optional[TabStaffSystem]:
        """
        Try to build a 6-line system, allowing for 1 missing line.
        
        If we find 5 peaks with one gap that's ~2x the normal spacing,
        we interpolate the missing line.
        """
        if start_idx + 4 >= len(peaks):
            return None
        
        # Try with 5 peaks (one might be missing)
        for num_peaks in [6, 5]:
            if start_idx + num_peaks - 1 >= len(peaks):
                continue
                
            candidate_peaks = peaks[start_idx:start_idx + num_peaks]
            
            if any(p in used_peaks for p in candidate_peaks):
                continue
            
            spacings = [candidate_peaks[i+1] - candidate_peaks[i] for i in range(len(candidate_peaks)-1)]
            
            if num_peaks == 5:
                # Check if one spacing is ~2x others (missing line)
                sorted_spacings = sorted(spacings)
                normal_spacing = sum(sorted_spacings[:3]) / 3  # Average of 3 smallest
                
                if normal_spacing < 5:
                    continue
                
                # Find the gap
                gap_idx = None
                for i, s in enumerate(spacings):
                    if 1.7 < s / normal_spacing < 2.3:  # ~2x normal
                        gap_idx = i
                        break
                
                if gap_idx is not None:
                    # Interpolate the missing line
                    interpolated_y = (candidate_peaks[gap_idx] + candidate_peaks[gap_idx + 1]) / 2
                    
                    # Insert into candidates
                    full_peaks = list(candidate_peaks)
                    full_peaks.insert(gap_idx + 1, int(interpolated_y))
                    
                    # Recalculate spacings
                    new_spacings = [full_peaks[i+1] - full_peaks[i] for i in range(5)]
                    avg_spacing = sum(new_spacings) / 5
                    
                    variance = max(abs(s - avg_spacing) / avg_spacing for s in new_spacings)
                    if variance < self.spacing_tolerance:
                        confidence = 0.8 * (1.0 - variance)  # Lower confidence for interpolated
                        
                        return TabStaffSystem(
                            line_y_positions=[float(p) for p in full_peaks],
                            line_spacing=avg_spacing,
                            y_start=full_peaks[0] - avg_spacing / 2,
                            y_end=full_peaks[5] + avg_spacing / 2,
                            confidence=confidence
                        )
        
        return None
    
    def detect_file(self, path: str) -> List[TabStaffSystem]:
        """Detect from file"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read: {path}")
        return self.detect(image)
    
    def visualize(self, image: np.ndarray, systems: List[TabStaffSystem]) -> np.ndarray:
        """Draw detected lines on image for verification"""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        height, width = result.shape[:2]
        
        colors = [
            (0, 0, 255),    # Red - String 1
            (0, 128, 255),  # Orange - String 2
            (0, 255, 255),  # Yellow - String 3
            (0, 255, 0),    # Green - String 4
            (255, 255, 0),  # Cyan - String 5
            (255, 0, 0),    # Blue - String 6
        ]
        
        for system in systems:
            for i, y in enumerate(system.line_y_positions):
                color = colors[i % 6]
                y_int = int(y)
                cv2.line(result, (0, y_int), (width, y_int), color, 2)
                cv2.putText(result, f"S{i+1}", (10, y_int - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    def get_profile_visualization(self, image: np.ndarray) -> np.ndarray:
        """Create visualization of the horizontal projection profile"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        height, width = binary.shape
        
        # Compute profile
        h_profile = np.sum(binary, axis=1) / 255
        
        # Normalize to fit in visualization
        max_val = max(h_profile)
        if max_val > 0:
            h_profile_norm = (h_profile / max_val * 200).astype(np.int32)
        else:
            h_profile_norm = h_profile.astype(np.int32)
        
        # Create visualization
        viz_width = 250
        viz = np.ones((height, viz_width, 3), dtype=np.uint8) * 255
        
        # Draw profile as horizontal bars
        for y in range(height):
            bar_width = h_profile_norm[y]
            cv2.line(viz, (0, y), (bar_width, y), (100, 100, 100), 1)
        
        # Draw threshold line
        threshold = int(width * self.min_line_width_ratio / max_val * 200)
        cv2.line(viz, (threshold, 0), (threshold, height), (0, 0, 255), 1)
        
        return viz


# CLI for testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python horizontal_projection.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Detecting TAB lines: {path}")
    print("=" * 60)
    
    detector = HorizontalProjection()
    image = cv2.imread(path)
    
    systems = detector.detect(image)
    
    print(f"\nFound {len(systems)} TAB systems")
    
    for i, system in enumerate(systems):
        print(f"\nSystem {i+1}:")
        print(f"  Confidence: {system.confidence:.1%}")
        print(f"  Line spacing: {system.line_spacing:.1f}px")
        print(f"  Y range: {system.y_start:.0f} - {system.y_end:.0f}")
        print(f"  Line Y positions:")
        for j, y in enumerate(system.line_y_positions):
            print(f"    String {j+1}: Y = {y:.0f}")
    
    # Save visualization
    viz = detector.visualize(image, systems)
    profile_viz = detector.get_profile_visualization(image)
    
    # Combine original + profile
    combined = np.hstack([viz, profile_viz])
    
    output_path = path.replace('.png', '_lines_detected.png')
    cv2.imwrite(output_path, combined)
    print(f"\nVisualization saved: {output_path}")
