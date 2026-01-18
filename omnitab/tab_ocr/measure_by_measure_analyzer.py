"""
Measure-by-Measure TAB Analyzer

Strategy:
1. Split image into systems using ScoreSlicer
2. Split each system into measures
3. Analyze each measure individually with Gemini
4. Combine results
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

from .preprocessor.score_slicer import ScoreSlicer


class MeasureByMeasureAnalyzer:
    """
    Analyze TAB by processing one measure at a time.
    
    This reduces complexity for Gemini and improves accuracy.
    """
    
    MEASURE_PROMPT = """Look at this SINGLE MEASURE of guitar TAB.

TAB READING RULES:
- 6 horizontal lines (strings 1-6, top to bottom)
- String 1 = TOP line, String 6 = BOTTOM line
- Numbers = fret positions
- Vertically aligned numbers = same beat

For this measure, identify EACH BEAT (vertically aligned group of numbers).

Output JSON:
{
  "beats": [
    {"notes": [{"string": 1, "fret": 10}, {"string": 2, "fret": 0}]},
    {"notes": [{"string": 1, "fret": 12}]}
  ]
}

RULES:
- Only include strings that have a number
- <12> = harmonic, use fret 12
- x/X = muted, use fret -1
- Count strings from TOP (1) to BOTTOM (6)

Analyze this measure:"""

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
        self.slicer = ScoreSlicer(output_dir="temp_slices")
    
    def extract_tab_portion(self, system_image: np.ndarray) -> np.ndarray:
        """Extract only TAB portion from system (lower half after staff)"""
        h, w = system_image.shape[:2]
        
        gray = cv2.cvtColor(system_image, cv2.COLOR_BGR2GRAY) if len(system_image.shape) == 3 else system_image
        
        # Find horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w*0.4, maxLineGap=20)
        
        if lines is None or len(lines) < 5:
            # Can't detect, return lower 60%
            return system_image[int(h*0.4):, :]
        
        # Find line Y positions
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                line_ys.append((y1 + y2) // 2)
        
        line_ys = sorted(set(line_ys))
        
        if len(line_ys) >= 11:
            # Staff (5) + TAB (6)
            tab_start = line_ys[5] - 15
        elif len(line_ys) >= 6:
            tab_start = line_ys[0] - 15
        else:
            tab_start = int(h * 0.4)
        
        tab_start = max(0, tab_start)
        return system_image[tab_start:, :]
    
    def split_into_measures(self, tab_image: np.ndarray) -> List[np.ndarray]:
        """Split TAB image into individual measures"""
        h, w = tab_image.shape[:2]
        gray = cv2.cvtColor(tab_image, cv2.COLOR_BGR2GRAY) if len(tab_image.shape) == 3 else tab_image
        
        # Detect vertical bar lines
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Vertical kernel to detect bar lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Sum columns
        col_sums = np.sum(vertical_lines, axis=0)
        
        # Find peaks (bar line positions)
        threshold = np.max(col_sums) * 0.4 if np.max(col_sums) > 0 else h * 0.3
        
        bar_positions = [0]
        in_peak = False
        peak_start = 0
        
        for i, val in enumerate(col_sums):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                in_peak = False
                bar_positions.append((peak_start + i) // 2)
        
        bar_positions.append(w)
        
        # Remove duplicates and sort
        bar_positions = sorted(set(bar_positions))
        
        # Filter: minimum measure width
        filtered = [bar_positions[0]]
        for pos in bar_positions[1:]:
            if pos - filtered[-1] > 50:  # At least 50px wide
                filtered.append(pos)
        
        # Split into measures
        measures = []
        for i in range(len(filtered) - 1):
            x1, x2 = filtered[i], filtered[i + 1]
            if x2 - x1 > 30:
                measure_img = tab_image[:, x1:x2]
                measures.append(measure_img)
        
        return measures
    
    def analyze_measure(self, measure_image: np.ndarray) -> Dict:
        """Analyze a single measure with Gemini"""
        _, buffer = cv2.imencode('.png', measure_image)
        image_data = buffer.tobytes()
        
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.MEASURE_PROMPT, image_part]
        )
        
        return self._parse_response(response.text)
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse JSON from response"""
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
            return {"beats": []}
    
    def analyze(self, image_path: str, debug: bool = False) -> Dict:
        """Full analysis pipeline"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        print("[M-by-M] Slicing into systems...")
        systems = self.slicer.slice_into_systems(image_path, save=False)
        print(f"[M-by-M] Found {len(systems)} systems")
        
        all_measures = []
        measure_num = 1
        
        for sys_idx, system in enumerate(systems):
            print(f"[M-by-M] Processing system {sys_idx + 1}...")
            
            # Extract TAB portion
            tab_img = self.extract_tab_portion(system.image)
            
            # Split into measures
            measures = self.split_into_measures(tab_img)
            print(f"[M-by-M]   Found {len(measures)} measures")
            
            for m_idx, measure_img in enumerate(measures):
                if debug:
                    cv2.imwrite(f"debug_m{measure_num}.png", measure_img)
                
                # Analyze measure
                result = self.analyze_measure(measure_img)
                
                beats = result.get("beats", [])
                print(f"[M-by-M]   Measure {measure_num}: {len(beats)} beats")
                
                if beats:
                    all_measures.append({
                        "number": measure_num,
                        "beats": beats
                    })
                
                measure_num += 1
        
        # Count notes
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
        print("Usage: python -m omnitab.tab_ocr.measure_by_measure_analyzer <image>")
        sys.exit(1)
    
    analyzer = MeasureByMeasureAnalyzer()
    result = analyzer.analyze(sys.argv[1], debug=True)
    
    print("\n=== Result ===")
    print(f"Measures: {len(result['measures'])}")
    print(f"Notes: {result['total_notes']}")
