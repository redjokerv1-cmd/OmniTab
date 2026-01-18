"""
Labeled TAB Analyzer - Add visual string labels to help Gemini

Strategy:
1. Detect TAB lines
2. Add "S1", "S2", ... "S6" labels to left side of each line
3. Analyze labeled image with Gemini
4. This helps Gemini understand which line is which string
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LabeledTabAnalyzer:
    """
    Add visual string labels to TAB image before analysis.
    
    This helps Gemini understand the string numbering.
    """
    
    LABELED_PROMPT = """Analyze this guitar TAB. I've added COLOR-CODED labels on the LEFT:

## COLOR LEGEND (look at colored boxes on left):
- RED "S1" = String 1 (top line)
- ORANGE "S2" = String 2
- YELLOW "S3" = String 3
- GREEN "S4" = String 4
- BLUE "S5" = String 5
- MAGENTA "S6" = String 6 (bottom line)

Each line also has a colored dot at the start matching its label color.

## HOW TO READ:
1. Look at a vertical column of numbers (same beat)
2. For each number, trace LEFT to find its colored label (S1-S6)
3. Record: string number from label, fret from the number

## EXAMPLE:
If column shows:
  10 (on line with RED S1 label)
   0 (on line with ORANGE S2 label)
  12 (on line with YELLOW S3 label)
  10 (on line with GREEN S4 label)

Then: {"notes": [{"string": 1, "fret": 10}, {"string": 2, "fret": 0}, {"string": 3, "fret": 12}, {"string": 4, "fret": 10}]}

## OUTPUT JSON:
{
  "measures": [
    {"number": 1, "beats": [
      {"notes": [{"string": 1, "fret": 10}, {"string": 2, "fret": 0}, {"string": 3, "fret": 12}, {"string": 4, "fret": 10}]}
    ]}
  ],
  "tuning": ["E", "C", "G", "D", "G", "C"],
  "capo": 2,
  "tempo": 65
}

## CRITICAL: Use the COLOR LABELS to identify strings. Don't guess from position alone.

Analyze:"""

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
    
    def detect_tab_lines(self, image: np.ndarray) -> List[int]:
        """Detect Y positions of 6 TAB lines with improved spacing validation"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                               minLineLength=w*0.4, maxLineGap=30)
        
        if lines is None:
            spacing = h // 7
            return [spacing * (i + 1) for i in range(6)]
        
        # Collect Y positions
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                line_ys.append((y1 + y2) // 2)
        
        # Cluster close lines
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
        
        # Find 6 consecutive evenly-spaced lines
        # TAB lines should be evenly spaced
        best_group = None
        best_variance = float('inf')
        
        for start in range(max(1, len(clustered) - 5)):
            group = clustered[start:start + 6]
            if len(group) < 6:
                continue
            
            # Calculate spacing variance
            spacings = [group[i+1] - group[i] for i in range(5)]
            avg_spacing = np.mean(spacings)
            
            # Check if spacing is reasonable (10-40 pixels typically)
            if avg_spacing < 8 or avg_spacing > 50:
                continue
            
            variance = np.var(spacings)
            
            if variance < best_variance:
                best_variance = variance
                best_group = group
        
        if best_group and best_variance < 30:  # Low variance = evenly spaced
            return best_group
        
        # Fallback: find last 6 lines that are roughly evenly spaced
        if len(clustered) >= 6:
            # Try from the end
            for start in range(len(clustered) - 6, -1, -1):
                group = clustered[start:start + 6]
                spacings = [group[i+1] - group[i] for i in range(5)]
                avg_spacing = np.mean(spacings)
                
                if 8 <= avg_spacing <= 50:
                    return group
        
        # Final fallback: divide equally
        spacing = h // 7
        return [spacing * (i + 1) for i in range(6)]
    
    def add_string_labels(self, image: np.ndarray, line_ys: List[int]) -> np.ndarray:
        """Add colored labels and highlight each TAB line for easy identification"""
        labeled = image.copy()
        h, w = labeled.shape[:2]
        
        # Colors for each string (BGR format) - distinct, easy to identify
        colors = [
            (0, 0, 255),     # Red for String 1
            (0, 165, 255),   # Orange for String 2
            (0, 255, 255),   # Yellow for String 3
            (0, 255, 0),     # Green for String 4
            (255, 0, 0),     # Blue for String 5
            (255, 0, 255),   # Magenta for String 6
        ]
        
        # Draw colored dots at the start of each line
        for i, y in enumerate(line_ys):
            color = colors[i % 6]
            # Draw colored circle at line start
            cv2.circle(labeled, (10, y), 6, color, -1)
            cv2.circle(labeled, (10, y), 6, (0, 0, 0), 1)  # Black outline
        
        # Create white area on left for labels
        padding = 50
        canvas = np.ones((h, w + padding, 3), dtype=np.uint8) * 255
        canvas[:, padding:] = labeled
        
        # Add colored labels with box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for i, y in enumerate(line_ys):
            color = colors[i % 6]
            label = f"S{i + 1}"
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            x = 5
            y_pos = y + text_size[1] // 2
            
            # Draw background rectangle
            cv2.rectangle(canvas, (x - 2, y_pos - text_size[1] - 2),
                         (x + text_size[0] + 2, y_pos + 4), color, -1)
            
            # Draw white text on colored background
            cv2.putText(canvas, label, (x, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        return canvas
    
    def detect_systems(self, image: np.ndarray) -> List[tuple]:
        """Detect systems (staff+TAB pairs)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Horizontal projection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_proj = np.sum(thresh, axis=1)
        
        # Find content regions
        threshold = w * 0.05
        in_region = False
        regions = []
        start_y = 0
        
        for i, val in enumerate(h_proj):
            if val > threshold and not in_region:
                in_region = True
                start_y = i
            elif val <= threshold and in_region:
                in_region = False
                if i - start_y > 80:
                    regions.append((max(0, start_y - 20), min(h, i + 20)))
        
        if in_region and h - start_y > 80:
            regions.append((max(0, start_y - 20), h))
        
        if not regions:
            regions = [(0, h)]
        
        return regions
    
    def extract_tab_from_system(self, system_img: np.ndarray) -> np.ndarray:
        """Extract TAB portion from system"""
        h, w = system_img.shape[:2]
        gray = cv2.cvtColor(system_img, cv2.COLOR_BGR2GRAY) if len(system_img.shape) == 3 else system_img
        
        # Find lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=w*0.3, maxLineGap=20)
        
        if lines is None or len(lines) < 5:
            return system_img[h//3:, :]
        
        line_ys = sorted([((l[0][1] + l[0][3]) // 2) for l in lines if abs(l[0][1] - l[0][3]) < 5])
        
        # Cluster
        clustered = []
        i = 0
        while i < len(line_ys):
            cluster = [line_ys[i]]
            j = i + 1
            while j < len(line_ys) and line_ys[j] - cluster[-1] < 10:
                cluster.append(line_ys[j])
                j += 1
            clustered.append(int(np.mean(cluster)))
            i = j
        
        if len(clustered) >= 11:
            # Staff + TAB
            tab_start = clustered[5] - 20
            tab_end = clustered[-1] + 30
        elif len(clustered) >= 6:
            tab_start = clustered[0] - 20
            tab_end = clustered[-1] + 30
        else:
            tab_start = h // 3
            tab_end = h
        
        return system_img[max(0, tab_start):min(h, tab_end), :]
    
    def analyze_labeled_image(self, labeled_image: np.ndarray) -> Dict:
        """Analyze labeled TAB image with Gemini"""
        _, buffer = cv2.imencode('.png', labeled_image)
        image_data = buffer.tobytes()
        
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.LABELED_PROMPT, image_part]
        )
        
        return self._parse_response(response.text)
    
    def _parse_response(self, response_text: str) -> Dict:
        try:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            
            return json.loads(json_str)
        except:
            return {"measures": [], "error": "parse failed"}
    
    def analyze(self, image_path: str, debug: bool = False) -> Dict:
        """Full analysis with labeled TAB lines"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        print("[Labeled] Detecting systems...")
        systems = self.detect_systems(image)
        print(f"[Labeled] Found {len(systems)} systems")
        
        all_measures = []
        measure_num = 1
        
        for sys_idx, (y_start, y_end) in enumerate(systems):
            system_img = image[y_start:y_end, :]
            
            print(f"[Labeled] Processing system {sys_idx + 1}...")
            
            # Extract TAB
            tab_img = self.extract_tab_from_system(system_img)
            
            # Detect TAB lines
            line_ys = self.detect_tab_lines(tab_img)
            print(f"[Labeled]   Lines at Y: {line_ys}")
            
            # Add labels
            labeled_img = self.add_string_labels(tab_img, line_ys)
            
            if debug:
                cv2.imwrite(f"debug_labeled_system_{sys_idx + 1}.png", labeled_img)
            
            # Analyze
            result = self.analyze_labeled_image(labeled_img)
            
            for m in result.get("measures", []):
                m["number"] = measure_num
                all_measures.append(m)
                measure_num += 1
            
            print(f"[Labeled]   Got {len(result.get('measures', []))} measures")
        
        # Total notes
        total_notes = sum(
            len(b.get("notes", []))
            for m in all_measures
            for b in m.get("beats", [])
        )
        
        return {
            "measures": all_measures,
            "total_notes": total_notes,
            "tuning": result.get("tuning"),
            "capo": result.get("capo"),
            "tempo": result.get("tempo")
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m omnitab.tab_ocr.labeled_tab_analyzer <image>")
        sys.exit(1)
    
    analyzer = LabeledTabAnalyzer()
    result = analyzer.analyze(sys.argv[1], debug=True)
    
    print("\n=== Result ===")
    print(f"Measures: {len(result['measures'])}")
    print(f"Notes: {result['total_notes']}")
