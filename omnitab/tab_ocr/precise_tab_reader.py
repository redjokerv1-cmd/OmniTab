"""
Precise TAB Reader - High accuracy TAB recognition

Strategy:
1. Detect TAB systems precisely using horizontal line detection
2. For each system, detect exact Y positions of 6 TAB lines
3. Use EasyOCR to get all numbers with precise X, Y coordinates
4. Map each number to nearest TAB line (string 1-6)
5. Group by X position to form beats
6. Detect measure boundaries
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TabLine:
    """A single TAB line (one of 6 strings)"""
    string_num: int  # 1-6
    y_position: int
    y_tolerance: int  # How close a digit must be to belong to this line


@dataclass
class DetectedNumber:
    """A number detected by OCR"""
    value: int
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class TabNote:
    """A note on the TAB"""
    string: int  # 1-6
    fret: int    # 0-24
    x_pos: int   # X coordinate


class PreciseTabReader:
    """
    High accuracy TAB reader using precise line detection + OCR
    """
    
    def __init__(self):
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr not installed")
        
        print("[PreciseTab] Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[PreciseTab] Ready")
    
    def detect_tab_systems(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect TAB systems in the image
        
        Returns list of (y_start, y_end) for each system
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=w*0.5, maxLineGap=30)
        
        if lines is None:
            return [(0, h)]
        
        # Collect Y positions of horizontal lines
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Nearly horizontal
                line_ys.append((y1 + y2) // 2)
        
        if not line_ys:
            return [(0, h)]
        
        line_ys = sorted(set(line_ys))
        
        # Find groups of 6+ consecutive lines (TAB systems)
        # Staff has 5 lines, TAB has 6 lines
        systems = []
        i = 0
        while i < len(line_ys) - 5:
            # Check if next 6 lines are roughly equally spaced
            group = line_ys[i:i+6]
            spacings = [group[j+1] - group[j] for j in range(5)]
            avg_spacing = np.mean(spacings)
            
            if avg_spacing > 5 and all(abs(s - avg_spacing) < avg_spacing * 0.5 for s in spacings):
                # Found a TAB system
                y_start = group[0] - 20
                y_end = group[5] + 20
                systems.append((max(0, y_start), min(h, y_end)))
                i += 6
            else:
                i += 1
        
        if not systems:
            # Fallback: divide image into equal parts
            num_systems = max(1, h // 200)
            part_h = h // num_systems
            systems = [(i * part_h, (i + 1) * part_h) for i in range(num_systems)]
        
        return systems
    
    def detect_tab_lines(self, image: np.ndarray) -> List[TabLine]:
        """
        Detect the 6 TAB lines in a system image
        
        Returns list of TabLine objects with precise Y positions
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines with Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                               minLineLength=w*0.3, maxLineGap=30)
        
        if lines is None or len(lines) < 6:
            # Fallback: use horizontal projection
            return self._detect_lines_with_projection(gray)
        
        # Collect Y positions
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                line_ys.append((y1 + y2) // 2)
        
        # Cluster close Y values
        line_ys = sorted(set(line_ys))
        clustered = []
        i = 0
        while i < len(line_ys):
            cluster = [line_ys[i]]
            j = i + 1
            while j < len(line_ys) and line_ys[j] - cluster[-1] < 8:
                cluster.append(line_ys[j])
                j += 1
            clustered.append(int(np.mean(cluster)))
            i = j
        
        # Take last 6 if we have more (skip staff lines)
        if len(clustered) >= 11:  # 5 staff + 6 TAB
            clustered = clustered[-6:]
        elif len(clustered) < 6:
            return self._detect_lines_with_projection(gray)
        
        clustered = clustered[:6]
        
        # Calculate tolerance (half of average spacing)
        spacings = [clustered[i+1] - clustered[i] for i in range(5)]
        tolerance = int(np.mean(spacings) / 2)
        
        tab_lines = []
        for i, y in enumerate(clustered):
            tab_lines.append(TabLine(
                string_num=i + 1,
                y_position=y,
                y_tolerance=tolerance
            ))
        
        return tab_lines
    
    def _detect_lines_with_projection(self, gray: np.ndarray) -> List[TabLine]:
        """Fallback line detection using horizontal projection"""
        h, w = gray.shape
        
        # Binary threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        h_proj = np.sum(thresh, axis=1)
        
        # Find peaks (lines have high values)
        threshold = np.max(h_proj) * 0.3
        peaks = []
        in_peak = False
        peak_start = 0
        
        for i, val in enumerate(h_proj):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                in_peak = False
                peak_center = (peak_start + i) // 2
                peaks.append(peak_center)
        
        # Take last 6 peaks (or divide equally if not enough)
        if len(peaks) >= 6:
            peaks = peaks[-6:]
        else:
            spacing = h // 7
            peaks = [spacing * (i + 1) for i in range(6)]
        
        tolerance = (peaks[1] - peaks[0]) // 2 if len(peaks) > 1 else 10
        
        return [TabLine(i + 1, y, tolerance) for i, y in enumerate(peaks)]
    
    def ocr_numbers(self, image: np.ndarray) -> List[DetectedNumber]:
        """
        Use EasyOCR to detect all numbers in the image
        """
        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # OCR
        results = self.reader.readtext(gray, 
                                       allowlist='0123456789',
                                       paragraph=False,
                                       min_size=5,
                                       width_ths=0.3)
        
        numbers = []
        for bbox, text, conf in results:
            if not text.isdigit():
                continue
            
            # Calculate center and size
            pts = np.array(bbox)
            x = int(np.mean([p[0] for p in pts]))
            y = int(np.mean([p[1] for p in pts]))
            width = int(max(p[0] for p in pts) - min(p[0] for p in pts))
            height = int(max(p[1] for p in pts) - min(p[1] for p in pts))
            
            numbers.append(DetectedNumber(
                value=int(text),
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=conf
            ))
        
        return numbers
    
    def map_to_strings(self, numbers: List[DetectedNumber], 
                       tab_lines: List[TabLine]) -> List[TabNote]:
        """
        Map each detected number to the nearest TAB line (string)
        """
        notes = []
        
        for num in numbers:
            # Find nearest line
            best_line = None
            best_dist = float('inf')
            
            for line in tab_lines:
                dist = abs(num.y - line.y_position)
                if dist < best_dist and dist <= line.y_tolerance * 2:
                    best_dist = dist
                    best_line = line
            
            if best_line:
                notes.append(TabNote(
                    string=best_line.string_num,
                    fret=num.value,
                    x_pos=num.x
                ))
        
        return notes
    
    def group_into_beats(self, notes: List[TabNote], 
                        x_threshold: int = 15) -> List[Dict]:
        """
        Group notes by X position to form beats
        """
        if not notes:
            return []
        
        sorted_notes = sorted(notes, key=lambda n: n.x_pos)
        
        beats = []
        current_beat = [sorted_notes[0]]
        current_x = sorted_notes[0].x_pos
        
        for note in sorted_notes[1:]:
            if note.x_pos - current_x <= x_threshold:
                current_beat.append(note)
            else:
                # New beat
                beats.append({
                    "notes": [{"string": n.string, "fret": n.fret} for n in current_beat]
                })
                current_beat = [note]
                current_x = note.x_pos
        
        # Last beat
        if current_beat:
            beats.append({
                "notes": [{"string": n.string, "fret": n.fret} for n in current_beat]
            })
        
        return beats
    
    def analyze(self, image_path: str, debug: bool = False) -> Dict:
        """
        Full analysis pipeline
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        h, w = image.shape[:2]
        
        print("[PreciseTab] Detecting TAB systems...")
        systems = self.detect_tab_systems(image)
        print(f"[PreciseTab] Found {len(systems)} systems")
        
        all_measures = []
        measure_num = 1
        
        for sys_idx, (y_start, y_end) in enumerate(systems):
            system_img = image[y_start:y_end, :]
            
            print(f"[PreciseTab] Processing system {sys_idx + 1}...")
            
            # Detect TAB lines
            tab_lines = self.detect_tab_lines(system_img)
            print(f"[PreciseTab]   Lines at Y: {[l.y_position for l in tab_lines]}")
            
            if debug:
                debug_img = system_img.copy()
                for line in tab_lines:
                    cv2.line(debug_img, (0, line.y_position), 
                            (w, line.y_position), (0, 255, 0), 1)
                cv2.imwrite(f"debug_system_{sys_idx + 1}_lines.png", debug_img)
            
            # OCR for numbers
            numbers = self.ocr_numbers(system_img)
            print(f"[PreciseTab]   OCR found {len(numbers)} numbers")
            
            # Map to strings
            notes = self.map_to_strings(numbers, tab_lines)
            print(f"[PreciseTab]   Mapped {len(notes)} notes")
            
            # Group into beats
            beats = self.group_into_beats(notes)
            print(f"[PreciseTab]   Grouped into {len(beats)} beats")
            
            # Create measures (for now, put all beats in one measure per system)
            if beats:
                all_measures.append({
                    "number": measure_num,
                    "beats": beats
                })
                measure_num += 1
        
        # Count total notes
        total_notes = sum(
            len(b.get("notes", []))
            for m in all_measures
            for b in m.get("beats", [])
        )
        
        return {
            "measures": all_measures,
            "total_notes": total_notes,
            "systems_analyzed": len(systems)
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m omnitab.tab_ocr.precise_tab_reader <image>")
        sys.exit(1)
    
    reader = PreciseTabReader()
    result = reader.analyze(sys.argv[1], debug=True)
    
    print("\n=== Result ===")
    print(f"Measures: {len(result['measures'])}")
    print(f"Notes: {result['total_notes']}")
    
    if result['measures']:
        print("\n=== First Measure Beats ===")
        m1 = result['measures'][0]
        for i, beat in enumerate(m1.get('beats', [])[:5]):
            notes = ', '.join([f"S{n['string']}:F{n['fret']}" for n in beat['notes']])
            print(f"Beat {i+1}: {notes}")
