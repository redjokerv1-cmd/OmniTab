"""
ScoreSlicer - Split score images into systems (lines) and measures

Uses morphological dilation to group musical elements into
detectable regions, then sorts and crops them in order.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SlicedRegion:
    """A sliced region from the score"""
    index: int
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    path: Optional[str] = None


class ScoreSlicer:
    """
    Split score images into systems (lines) or measures
    
    Uses OpenCV morphological operations to detect and isolate
    musical content regions.
    """
    
    def __init__(self, output_dir: str = "sliced"):
        """
        Initialize ScoreSlicer
        
        Args:
            output_dir: Directory to save sliced images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def slice_into_systems(
        self,
        image_path: str,
        kernel_width: int = 100,
        kernel_height: int = 30,
        min_width_ratio: float = 0.5,
        min_height: int = 50,
        padding: int = 20,
        save: bool = True
    ) -> List[SlicedRegion]:
        """
        Slice score image into systems (horizontal lines of music)
        
        This is ideal for processing each staff system separately,
        which improves Gemini's recognition accuracy.
        
        Args:
            image_path: Path to score image
            kernel_width: Dilation kernel width (larger = merge more horizontally)
            kernel_height: Dilation kernel height (larger = merge staff + TAB)
            min_width_ratio: Minimum width as ratio of image width
            min_height: Minimum height in pixels
            padding: Padding around cropped region
            save: Whether to save cropped images
            
        Returns:
            List of SlicedRegion objects
        """
        logger.info(f"[Slicer] Slicing into systems: {image_path}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold (invert: black on white -> white on black)
        _, thresh = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological dilation - key step!
        # Merges scattered notes/lines into solid rectangular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                           (kernel_width, kernel_height))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by Y coordinate (top to bottom)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # Extract regions
        regions = []
        system_idx = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter noise
            if w < img_w * min_width_ratio:
                continue
            if h < min_height:
                continue
            
            # Crop with padding
            y1 = max(0, y - padding)
            y2 = min(img_h, y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(img_w, x + w + padding)
            
            cropped = img[y1:y2, x1:x2]
            
            region = SlicedRegion(
                index=system_idx,
                image=cropped,
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1
            )
            
            # Save if requested
            if save:
                filename = self.output_dir / f"system_{system_idx:03d}.png"
                cv2.imwrite(str(filename), cropped)
                region.path = str(filename)
                logger.debug(f"[Slicer] Saved: {filename}")
            
            regions.append(region)
            system_idx += 1
        
        logger.info(f"[Slicer] Found {len(regions)} systems")
        return regions
    
    def slice_into_measures(
        self,
        image_path: str,
        min_width: int = 50,
        padding: int = 10,
        save: bool = True
    ) -> List[SlicedRegion]:
        """
        Slice score image into individual measures
        
        Uses vertical projection to detect bar lines and split measures.
        Enhanced algorithm that looks for strong vertical lines.
        
        Args:
            image_path: Path to score image (or system slice)
            min_width: Minimum measure width in pixels
            padding: Padding around cropped region
            save: Whether to save cropped images
            
        Returns:
            List of SlicedRegion objects
        """
        logger.info(f"[Slicer] Slicing into measures: {image_path}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect vertical lines (bar lines)
        # Use a tall, narrow kernel to detect vertical structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_h // 3))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Vertical projection
        vertical_proj = np.sum(vertical_lines, axis=0)
        
        # Find peaks (bar line positions)
        # A bar line will have high sum in its column
        if vertical_proj.max() > 0:
            threshold = vertical_proj.max() * 0.5
        else:
            threshold = img_h * 0.3
        
        bar_lines = []
        in_peak = False
        peak_start = 0
        
        for i, val in enumerate(vertical_proj):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                in_peak = False
                peak_center = (peak_start + i) // 2
                bar_lines.append(peak_center)
        
        logger.info(f"[Slicer] Detected {len(bar_lines)} bar lines")
        
        # If too few bar lines, try alternative method
        if len(bar_lines) < 2:
            logger.info("[Slicer] Few bar lines, trying edge detection")
            bar_lines = self._detect_bar_lines_with_edges(gray, img_h)
        
        # If still no bar lines, split evenly
        if len(bar_lines) < 2:
            logger.info("[Slicer] No bar lines found, splitting evenly")
            num_measures = max(1, img_w // 200)  # Assume ~200px per measure
            bar_lines = [int(img_w * i / num_measures) for i in range(num_measures + 1)]
        
        # Add boundaries
        if bar_lines[0] > 20:
            bar_lines = [0] + bar_lines
        if bar_lines[-1] < img_w - 20:
            bar_lines = bar_lines + [img_w]
        
        # Extract measures
        regions = []
        measure_idx = 0
        
        for i in range(len(bar_lines) - 1):
            x1 = max(0, bar_lines[i])
            x2 = min(img_w, bar_lines[i + 1])
            
            # Skip too narrow
            if x2 - x1 < min_width:
                continue
            
            # Add padding
            x1_pad = max(0, x1 - padding)
            x2_pad = min(img_w, x2 + padding)
            
            cropped = img[0:img_h, x1_pad:x2_pad]
            
            region = SlicedRegion(
                index=measure_idx,
                image=cropped,
                x=x1_pad,
                y=0,
                width=x2_pad - x1_pad,
                height=img_h
            )
            
            if save:
                filename = self.output_dir / f"measure_{measure_idx:03d}.png"
                cv2.imwrite(str(filename), cropped)
                region.path = str(filename)
            
            regions.append(region)
            measure_idx += 1
        
        logger.info(f"[Slicer] Found {len(regions)} measures")
        return regions
    
    def _detect_bar_lines_with_edges(
        self,
        gray: np.ndarray,
        img_h: int
    ) -> List[int]:
        """Detect bar lines using edge detection"""
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=img_h // 3,
                                minLineLength=img_h // 2,
                                maxLineGap=10)
        
        if lines is None:
            return []
        
        # Filter vertical lines
        vertical_xs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is vertical (small x difference)
            if abs(x2 - x1) < 5:
                vertical_xs.append((x1 + x2) // 2)
        
        # Cluster nearby x values
        if not vertical_xs:
            return []
        
        vertical_xs = sorted(vertical_xs)
        clustered = [vertical_xs[0]]
        
        for x in vertical_xs[1:]:
            if x - clustered[-1] > 20:  # Gap threshold
                clustered.append(x)
        
        return clustered
    
    def _slice_measures_by_dilation(
        self,
        img: np.ndarray,
        thresh: np.ndarray,
        min_width: int,
        min_height_ratio: float,
        padding: int,
        save: bool
    ) -> List[SlicedRegion]:
        """Fallback: slice measures using dilation"""
        img_h, img_w = img.shape[:2]
        
        # Vertical dilation kernel (merge vertically, separate horizontally)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by X (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        regions = []
        measure_idx = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < min_width:
                continue
            if h < img_h * min_height_ratio:
                continue
            
            x1 = max(0, x - padding)
            x2 = min(img_w, x + w + padding)
            y1 = max(0, y - padding)
            y2 = min(img_h, y + h + padding)
            
            cropped = img[y1:y2, x1:x2]
            
            region = SlicedRegion(
                index=measure_idx,
                image=cropped,
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1
            )
            
            if save:
                filename = self.output_dir / f"measure_{measure_idx:03d}.png"
                cv2.imwrite(str(filename), cropped)
                region.path = str(filename)
            
            regions.append(region)
            measure_idx += 1
        
        return regions
    
    def visualize_slices(
        self,
        image_path: str,
        regions: List[SlicedRegion],
        output_path: str = "slices_visualization.png"
    ):
        """Draw rectangles around detected regions for debugging"""
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for region in regions:
            color = colors[region.index % len(colors)]
            cv2.rectangle(img,
                         (region.x, region.y),
                         (region.x + region.width, region.y + region.height),
                         color, 2)
            cv2.putText(img, f"{region.index}",
                       (region.x + 5, region.y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imwrite(output_path, img)
        logger.info(f"[Slicer] Visualization saved: {output_path}")


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m omnitab.tab_ocr.preprocessor.score_slicer <image>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    slicer = ScoreSlicer()
    
    print("=== System Slicing ===")
    systems = slicer.slice_into_systems(sys.argv[1])
    print(f"Found {len(systems)} systems")
    
    if systems:
        print("\n=== Measure Slicing (first system) ===")
        measures = slicer.slice_into_measures(systems[0].path)
        print(f"Found {len(measures)} measures in system 0")
    
    # Visualize
    slicer.visualize_slices(sys.argv[1], systems, "systems_debug.png")
    print("\nVisualization saved: systems_debug.png")
