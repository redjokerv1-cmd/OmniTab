"""
Calibrated TAB Analyzer - Use first beat as calibration reference

Key insight: Providing the first beat as a "calibration example" helps
Gemini understand the string numbering for the rest of the analysis.

Strategy:
1. For each system, crop the TAB area precisely
2. Provide first beat answer as calibration in prompt
3. Ask Gemini to analyze remaining beats using that reference
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


class CalibratedAnalyzer:
    """
    Analyze TAB with calibration reference.
    
    The first beat of the first system is provided as a "calibration"
    to help Gemini understand string numbering.
    """
    
    CALIBRATED_PROMPT = """Analyze this guitar TAB image.

## CALIBRATION - First Beat Reference:
The FIRST beat in this TAB is: {calibration}
Use this to understand which line corresponds to which string number.

## TAB Structure:
- 6 horizontal lines (String 1=top to String 6=bottom)
- Numbers = fret positions
- Vertically aligned numbers = same beat
- Vertical bars | = measure boundaries

## Your Task:
Analyze ALL beats in this TAB, starting from the first beat (which I gave you as calibration).
Be consistent with the calibration - if S3 had fret 12 at a certain line, other S3 notes will be on the same line.

## Output JSON:
{{
  "measures": [
    {{
      "number": 1,
      "beats": [
        {{"notes": [{{"string": 1, "fret": 10}}, {{"string": 2, "fret": 0}}]}}
      ]
    }}
  ],
  "tuning": ["E", "C", "G", "D", "G", "C"],
  "capo": 2,
  "tempo": 65
}}

## Rules:
- String 1 = top line, String 6 = bottom line
- <12> = harmonic at fret 12
- x = muted note (fret: -1)
- Only include strings that have numbers

Analyze:"""

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
    
    def detect_systems(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Detect systems (staff+TAB pairs)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_proj = np.sum(thresh, axis=1)
        
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
                    regions.append((max(0, start_y - 10), min(h, i + 10)))
        
        if in_region and h - start_y > 80:
            regions.append((max(0, start_y - 10), h))
        
        return regions if regions else [(0, h)]
    
    def extract_tab_region(self, system_img: np.ndarray) -> np.ndarray:
        """Extract only TAB portion from system image"""
        h, w = system_img.shape[:2]
        gray = cv2.cvtColor(system_img, cv2.COLOR_BGR2GRAY) if len(system_img.shape) == 3 else system_img
        
        # Detect horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w*0.4, maxLineGap=30)
        
        if lines is None or len(lines) < 6:
            # Fallback: return lower 55% of image
            return system_img[int(h*0.45):, :]
        
        # Collect line Y positions
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                line_ys.append((y1 + y2) // 2)
        
        line_ys = sorted(set(line_ys))
        
        # Cluster
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
        
        # Find 6 consecutive evenly-spaced lines (TAB)
        best_group = None
        best_variance = float('inf')
        
        for start in range(max(1, len(clustered) - 5)):
            group = clustered[start:start + 6]
            if len(group) < 6:
                continue
            
            spacings = [group[i+1] - group[i] for i in range(5)]
            avg_spacing = np.mean(spacings)
            
            if avg_spacing < 8 or avg_spacing > 50:
                continue
            
            variance = np.var(spacings)
            if variance < best_variance:
                best_variance = variance
                best_group = group
        
        if best_group and best_variance < 30:
            y_start = max(0, best_group[0] - 25)
            y_end = min(h, best_group[-1] + 35)
            return system_img[y_start:y_end, :]
        
        # Fallback
        return system_img[int(h*0.45):, :]
    
    def analyze_with_calibration(self, image: np.ndarray, calibration: str) -> Dict:
        """Analyze TAB with calibration reference"""
        _, buffer = cv2.imencode('.png', image)
        image_data = buffer.tobytes()
        
        prompt = self.CALIBRATED_PROMPT.format(calibration=calibration)
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image_part]
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
    
    def analyze(self, image_path: str, 
                calibration: str = "S1:10, S2:0, S3:12, S4:10",
                debug: bool = False) -> Dict:
        """
        Analyze TAB with calibration.
        
        Args:
            image_path: Path to TAB image
            calibration: First beat as "S1:fret, S2:fret, ..." format
            debug: Save debug images
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        h, w = image.shape[:2]
        
        print("[Calibrated] Detecting systems...")
        systems = self.detect_systems(image)
        print(f"[Calibrated] Found {len(systems)} systems")
        
        all_measures = []
        measure_num = 1
        
        for sys_idx, (y_start, y_end) in enumerate(systems):
            system_img = image[y_start:y_end, :]
            
            print(f"[Calibrated] Processing system {sys_idx + 1}...")
            
            # Extract TAB region
            tab_img = self.extract_tab_region(system_img)
            
            if debug:
                cv2.imwrite(f"debug_calibrated_system_{sys_idx + 1}.png", tab_img)
            
            # Use calibration only for first system
            cal = calibration if sys_idx == 0 else "Continue from previous pattern"
            
            # Analyze
            result = self.analyze_with_calibration(tab_img, cal)
            
            for m in result.get("measures", []):
                m["number"] = measure_num
                all_measures.append(m)
                measure_num += 1
            
            print(f"[Calibrated]   Got {len(result.get('measures', []))} measures")
        
        # Count notes
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
        print("Usage: python -m omnitab.tab_ocr.calibrated_analyzer <image> [calibration]")
        print("Example calibration: 'S1:10, S2:0, S3:12, S4:10'")
        sys.exit(1)
    
    cal = sys.argv[2] if len(sys.argv) > 2 else "S1:10, S2:0, S3:12, S4:10"
    
    analyzer = CalibratedAnalyzer()
    result = analyzer.analyze(sys.argv[1], calibration=cal, debug=True)
    
    print("\n=== Result ===")
    print(f"Measures: {len(result['measures'])}")
    print(f"Notes: {result['total_notes']}")
