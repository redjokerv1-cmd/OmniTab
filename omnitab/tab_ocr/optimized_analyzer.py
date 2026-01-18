"""
Optimized TAB Analyzer - Best practices combined

Key insights from testing:
1. Extract TAB region precisely (exclude staff notation)
2. Split into smaller sections (measures)
3. Provide calibration with first beat
4. Use detailed prompt with expected structure
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


class OptimizedAnalyzer:
    """
    Optimized TAB analyzer combining all best practices.
    """
    
    SECTION_PROMPT = """Analyze this section of guitar TAB.

CALIBRATION (use this to understand string numbering):
First beat of this piece = {calibration}

TAB Structure:
- 6 horizontal lines (String 1=top, String 6=bottom)
- Numbers on lines = fret positions
- Vertically aligned numbers = same beat (played together)
- | vertical bars = measure boundaries

HOW TO READ:
1. Start from left side
2. Find each vertical column of numbers (same beat)
3. For each number, count which line it's on (1=top to 6=bottom)
4. Record: string number + fret number

EXAMPLE from calibration:
If first beat shows numbers 10, 0, 12, 10 stacked vertically:
- 10 on line 1 = String 1, fret 10
- 0 on line 2 = String 2, fret 0
- 12 on line 3 = String 3, fret 12
- 10 on line 4 = String 4, fret 10

Output JSON:
{{
  "beats": [
    {{"notes": [{{"string": 1, "fret": 10}}, {{"string": 2, "fret": 0}}, {{"string": 3, "fret": 12}}, {{"string": 4, "fret": 10}}]}},
    {{"notes": [{{"string": 1, "fret": 12}}, {{"string": 3, "fret": 14}}]}}
  ]
}}

RULES:
- Count lines from TOP (1) to BOTTOM (6)
- Only include strings that have a number
- <12> = harmonic at fret 12
- x = muted, use fret: -1
- Empty line = no note on that string

Be precise with line counting. Analyze:"""

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
    
    def detect_systems(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Detect systems"""
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
    
    def extract_tab_only(self, system_img: np.ndarray) -> np.ndarray:
        """Extract TAB portion only"""
        h, w = system_img.shape[:2]
        gray = cv2.cvtColor(system_img, cv2.COLOR_BGR2GRAY) if len(system_img.shape) == 3 else system_img
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w*0.4, maxLineGap=30)
        
        if lines is None:
            return system_img[int(h*0.45):, :]
        
        line_ys = sorted([((l[0][1] + l[0][3]) // 2) for l in lines if abs(l[0][1] - l[0][3]) < 5])
        
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
        
        # Find 6 evenly-spaced lines
        for start in range(len(clustered) - 5):
            group = clustered[start:start + 6]
            spacings = [group[i+1] - group[i] for i in range(5)]
            avg_spacing = np.mean(spacings)
            
            if 8 <= avg_spacing <= 50 and np.var(spacings) < 30:
                y_start = max(0, group[0] - 25)
                y_end = min(h, group[-1] + 35)
                return system_img[y_start:y_end, :]
        
        return system_img[int(h*0.45):, :]
    
    def split_into_sections(self, tab_img: np.ndarray, section_width: int = 300) -> List[np.ndarray]:
        """Split TAB into manageable sections"""
        h, w = tab_img.shape[:2]
        
        # Detect vertical bars (measure boundaries)
        gray = cv2.cvtColor(tab_img, cv2.COLOR_BGR2GRAY) if len(tab_img.shape) == 3 else tab_img
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        col_sums = np.sum(vertical_lines, axis=0)
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
                pos = (peak_start + i) // 2
                if pos - bar_positions[-1] > 50:
                    bar_positions.append(pos)
        
        bar_positions.append(w)
        
        # Split into sections (1-2 measures each)
        sections = []
        for i in range(len(bar_positions) - 1):
            x1, x2 = bar_positions[i], bar_positions[i + 1]
            if x2 - x1 > 30:
                sections.append(tab_img[:, x1:x2])
        
        return sections if sections else [tab_img]
    
    def analyze_section(self, section_img: np.ndarray, calibration: str) -> Dict:
        """Analyze a single section"""
        _, buffer = cv2.imencode('.png', section_img)
        image_data = buffer.tobytes()
        
        prompt = self.SECTION_PROMPT.format(calibration=calibration)
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image_part]
        )
        
        return self._parse_response(response.text)
    
    def _parse_response(self, text: str) -> Dict:
        try:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
            
            return json.loads(json_str)
        except:
            return {"beats": []}
    
    def analyze(self, image_path: str, 
                calibration: str = "S1:10, S2:0, S3:12, S4:10",
                debug: bool = False) -> Dict:
        """Full analysis"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        print("[Optimized] Detecting systems...")
        systems = self.detect_systems(image)
        print(f"[Optimized] Found {len(systems)} systems")
        
        all_measures = []
        measure_num = 1
        total_beats = []
        
        for sys_idx, (y_start, y_end) in enumerate(systems):
            system_img = image[y_start:y_end, :]
            
            print(f"[Optimized] Processing system {sys_idx + 1}...")
            
            # Extract TAB only
            tab_img = self.extract_tab_only(system_img)
            
            if debug:
                cv2.imwrite(f"debug_opt_system_{sys_idx + 1}.png", tab_img)
            
            # Split into sections
            sections = self.split_into_sections(tab_img)
            print(f"[Optimized]   Split into {len(sections)} sections")
            
            for sec_idx, section_img in enumerate(sections):
                if debug:
                    cv2.imwrite(f"debug_opt_s{sys_idx + 1}_sec{sec_idx + 1}.png", section_img)
                
                # Analyze section
                result = self.analyze_section(section_img, calibration)
                beats = result.get("beats", [])
                
                if beats:
                    total_beats.extend(beats)
                    all_measures.append({
                        "number": measure_num,
                        "beats": beats
                    })
                    measure_num += 1
                
                print(f"[Optimized]     Section {sec_idx + 1}: {len(beats)} beats")
        
        # Count notes
        total_notes = sum(len(b.get("notes", [])) for b in total_beats)
        
        return {
            "measures": all_measures,
            "total_notes": total_notes,
            "total_beats": len(total_beats)
        }


if __name__ == "__main__":
    import sys
    
    cal = "S1:10, S2:0, S3:12, S4:10"
    path = sys.argv[1] if len(sys.argv) > 1 else "test_samples/images/page_1.png"
    
    analyzer = OptimizedAnalyzer()
    result = analyzer.analyze(path, calibration=cal, debug=True)
    
    print("\n=== Result ===")
    print(f"Measures: {len(result['measures'])}")
    print(f"Notes: {result['total_notes']}")
    print(f"Beats: {result['total_beats']}")
