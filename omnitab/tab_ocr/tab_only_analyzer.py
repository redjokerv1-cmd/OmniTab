"""
TAB-Only Analyzer - Extract and analyze only the TAB portion of sheet music

Strategy:
1. Detect systems (staff + TAB pairs)
2. For each system, extract ONLY the TAB portion (lower half)
3. Analyze each TAB region with Gemini
4. Combine results
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


class TabOnlyAnalyzer:
    """
    Analyze TAB images by:
    1. Splitting into systems
    2. Extracting TAB portion only (excluding staff notation)
    3. Analyzing each TAB with focused prompt
    """
    
    TAB_ANALYSIS_PROMPT = """Analyze this guitar TAB image. This shows ONLY the TAB notation (6 horizontal lines with numbers).

## TAB Structure:
- 6 horizontal lines
- Top line = String 1 (high E or alternate tuning)
- Bottom line = String 6 (low E or alternate tuning)
- Numbers on lines = fret positions
- Numbers vertically aligned = play together (same beat)
- Vertical bars | = measure boundaries

## Your Task:
1. Identify measure boundaries (vertical lines that span all 6 strings)
2. Within each measure, identify each beat (vertically aligned group of numbers)
3. For each beat, list which string has which fret number

## Output Format (JSON):
{
  "measures": [
    {
      "number": 1,
      "beats": [
        {
          "notes": [
            {"string": 1, "fret": 10},
            {"string": 2, "fret": 0},
            {"string": 3, "fret": 12},
            {"string": 4, "fret": 10}
          ]
        },
        {
          "notes": [
            {"string": 1, "fret": 12},
            {"string": 3, "fret": 14}
          ]
        }
      ]
    }
  ]
}

## Rules:
1. Only include strings that have a number - skip empty positions
2. <12> means harmonic at fret 12 - use fret: 12
3. x or X = muted note, use fret: -1
4. Carefully count vertical alignment - numbers at same X position = same beat
5. H, P, AH, etc. are techniques - ignore for fret detection

Analyze this TAB:"""

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
    
    def detect_systems(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect systems (staff+TAB pairs) in the image
        
        Returns list of (x, y, w, h) bounding boxes
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection to find systems
        h_proj = np.sum(thresh, axis=1)
        
        # Find regions with content
        threshold = w * 0.1
        in_region = False
        regions = []
        start_y = 0
        
        for i, val in enumerate(h_proj):
            if val > threshold and not in_region:
                in_region = True
                start_y = i
            elif val <= threshold and in_region:
                in_region = False
                if i - start_y > 80:  # Min height for a system
                    regions.append((0, start_y, w, i - start_y))
        
        if in_region and h - start_y > 80:
            regions.append((0, start_y, w, h - start_y))
        
        return regions
    
    def extract_tab_portion(self, system_image: np.ndarray) -> np.ndarray:
        """
        Extract only the TAB portion from a system image
        
        The TAB is typically in the lower half of the system,
        below the 5-line staff notation.
        """
        h, w = system_image.shape[:2]
        gray = cv2.cvtColor(system_image, cv2.COLOR_BGR2GRAY) if len(system_image.shape) == 3 else system_image
        
        # Find horizontal lines (staff lines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w*0.5, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            # Can't detect lines, use lower half
            return system_image[h//2:, :]
        
        # Find line Y positions
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Horizontal lines only
            if abs(y2 - y1) < 5:
                line_ys.append((y1 + y2) // 2)
        
        line_ys = sorted(set(line_ys))
        
        # Group lines into staff (5) and TAB (6)
        # TAB is usually below staff
        if len(line_ys) >= 11:
            # Both staff and TAB detected
            # Staff: first 5, TAB: lines 6-11
            tab_start = line_ys[5] - 10
            tab_end = line_ys[-1] + 30
        elif len(line_ys) >= 6:
            # Only TAB or mixed
            tab_start = line_ys[0] - 10
            tab_end = line_ys[-1] + 30
        else:
            # Can't detect, use lower portion
            tab_start = h // 2
            tab_end = h
        
        tab_start = max(0, tab_start)
        tab_end = min(h, tab_end)
        
        return system_image[tab_start:tab_end, :]
    
    def analyze_tab_image(self, image: np.ndarray) -> Dict:
        """Analyze a TAB-only image with Gemini"""
        # Encode image
        _, buffer = cv2.imencode('.png', image)
        image_data = buffer.tobytes()
        
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.TAB_ANALYSIS_PROMPT, image_part]
        )
        
        return self._parse_response(response.text)
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse JSON from Gemini response"""
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
        except json.JSONDecodeError as e:
            return {"error": str(e), "raw": response_text[:500]}
    
    def analyze(self, image_path: str, debug: bool = False) -> Dict:
        """
        Full analysis pipeline:
        1. Load image
        2. Detect systems
        3. Extract TAB from each system
        4. Analyze each TAB
        5. Combine results
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        print("[TabOnly] Detecting systems...")
        systems = self.detect_systems(image)
        print(f"[TabOnly] Found {len(systems)} systems")
        
        all_measures = []
        measure_number = 1
        
        for i, (x, y, w, h) in enumerate(systems):
            system_img = image[y:y+h, x:x+w]
            
            print(f"[TabOnly] Processing system {i+1}...")
            tab_img = self.extract_tab_portion(system_img)
            
            if debug:
                cv2.imwrite(f"debug_system_{i+1}_tab.png", tab_img)
            
            result = self.analyze_tab_image(tab_img)
            
            if "error" in result:
                print(f"[TabOnly] Error in system {i+1}: {result['error']}")
                continue
            
            for m in result.get("measures", []):
                m["number"] = measure_number
                all_measures.append(m)
                measure_number += 1
            
            print(f"[TabOnly] System {i+1}: {len(result.get('measures', []))} measures")
        
        return {
            "measures": all_measures,
            "systems_analyzed": len(systems)
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m omnitab.tab_ocr.tab_only_analyzer <image>")
        sys.exit(1)
    
    analyzer = TabOnlyAnalyzer()
    result = analyzer.analyze(sys.argv[1], debug=True)
    
    print("\n=== Result ===")
    total_notes = sum(
        len(b.get("notes", []))
        for m in result.get("measures", [])
        for b in m.get("beats", [])
    )
    print(f"Measures: {len(result.get('measures', []))}")
    print(f"Notes: {total_notes}")
