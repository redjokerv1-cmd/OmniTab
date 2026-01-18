"""
Line-by-Line TAB Analyzer

Strategy:
1. Detect TAB region
2. Split into 6 horizontal strips (one per string)
3. Analyze each strip with Gemini (just numbers and X positions)
4. Combine results by X coordinate to form beats
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

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


@dataclass
class NotePosition:
    """A note with its X position"""
    string: int
    fret: int
    x_pos: float  # Relative X position (0-1)


class LineByLineAnalyzer:
    """
    Analyze TAB by reading each line separately.
    
    This eliminates string confusion by analyzing each line in isolation.
    """
    
    LINE_ANALYSIS_PROMPT = """Look at this image of a single horizontal line from guitar TAB.

This line represents STRING {string_num} of the guitar.

Your task:
1. Find all NUMBERS on this line (including 2-digit numbers like 10, 12, 15)
2. Note the X position of each number (as a percentage from left 0% to right 100%)
3. Also identify any special markers: <12> (harmonic), x or X (muted)

Return JSON format:
{{
  "frets": [
    {{"fret": 10, "x_percent": 5}},
    {{"fret": 12, "x_percent": 15}},
    {{"fret": 0, "x_percent": 25}},
    {{"fret": -1, "x_percent": 50, "type": "muted"}}
  ]
}}

Rules:
- Include ALL numbers you can see
- x or X = muted note, use fret: -1
- <12> = harmonic at fret 12
- If no numbers on this line, return {{"frets": []}}
- X position should be estimated from left (0) to right (100)

Analyze this line:"""

    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
    
    def detect_tab_lines(self, image: np.ndarray) -> List[int]:
        """
        Detect the Y positions of 6 TAB lines
        
        Returns list of 6 Y coordinates (from top to bottom = string 1-6)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=w*0.6, maxLineGap=20)
        
        if lines is None:
            # Fallback: divide equally
            spacing = h // 7
            return [spacing * (i + 1) for i in range(6)]
        
        # Collect Y positions of horizontal lines
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Nearly horizontal
                line_ys.append((y1 + y2) // 2)
        
        if len(line_ys) < 6:
            # Not enough lines detected
            spacing = h // 7
            return [spacing * (i + 1) for i in range(6)]
        
        # Cluster Y positions
        line_ys = sorted(set(line_ys))
        
        # Group nearby lines (within 5 pixels)
        grouped = []
        current_group = [line_ys[0]]
        
        for y in line_ys[1:]:
            if y - current_group[-1] < 10:
                current_group.append(y)
            else:
                grouped.append(int(np.mean(current_group)))
                current_group = [y]
        grouped.append(int(np.mean(current_group)))
        
        # Take last 6 if we have more (TAB is usually after staff)
        if len(grouped) > 6:
            grouped = grouped[-6:]
        elif len(grouped) < 6:
            # Interpolate missing lines
            spacing = h // 7
            grouped = [spacing * (i + 1) for i in range(6)]
        
        return grouped[:6]
    
    def extract_line_strip(self, image: np.ndarray, y_center: int, 
                          strip_height: int = 25) -> np.ndarray:
        """Extract a horizontal strip centered on y_center"""
        h, w = image.shape[:2]
        y1 = max(0, y_center - strip_height // 2)
        y2 = min(h, y_center + strip_height // 2)
        return image[y1:y2, :]
    
    def analyze_line(self, line_image: np.ndarray, string_num: int) -> List[Dict]:
        """Analyze a single TAB line image"""
        _, buffer = cv2.imencode('.png', line_image)
        image_data = buffer.tobytes()
        
        prompt = self.LINE_ANALYSIS_PROMPT.format(string_num=string_num)
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image_part]
        )
        
        result = self._parse_response(response.text)
        return result.get("frets", [])
    
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
            return {"frets": []}
    
    def group_into_beats(self, all_notes: List[NotePosition], 
                        x_threshold: float = 3.0) -> List[Dict]:
        """
        Group notes by X position to form beats
        
        Notes within x_threshold percent of each other = same beat
        """
        if not all_notes:
            return []
        
        # Sort by X position
        sorted_notes = sorted(all_notes, key=lambda n: n.x_pos)
        
        beats = []
        current_beat_notes = [sorted_notes[0]]
        current_x = sorted_notes[0].x_pos
        
        for note in sorted_notes[1:]:
            if note.x_pos - current_x <= x_threshold:
                current_beat_notes.append(note)
            else:
                # New beat
                beats.append({
                    "notes": [{"string": n.string, "fret": n.fret} 
                             for n in current_beat_notes]
                })
                current_beat_notes = [note]
                current_x = note.x_pos
        
        # Last beat
        if current_beat_notes:
            beats.append({
                "notes": [{"string": n.string, "fret": n.fret} 
                         for n in current_beat_notes]
            })
        
        return beats
    
    def detect_measure_boundaries(self, image: np.ndarray, 
                                  line_ys: List[int]) -> List[float]:
        """
        Detect vertical measure lines
        
        Returns list of X positions (as percentages) where measures start
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Focus on TAB region
        y_start = max(0, line_ys[0] - 20)
        y_end = min(h, line_ys[-1] + 20)
        tab_region = gray[y_start:y_end, :]
        
        # Detect vertical lines
        edges = cv2.Canny(tab_region, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                               minLineLength=(y_end - y_start) * 0.7,
                               maxLineGap=10)
        
        if lines is None:
            return [0, 100]  # Just start and end
        
        # Collect X positions of vertical lines
        measure_xs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:  # Nearly vertical
                x_percent = ((x1 + x2) / 2) / w * 100
                measure_xs.append(x_percent)
        
        # Group nearby lines
        measure_xs = sorted(set([int(x) for x in measure_xs]))
        
        if not measure_xs:
            return [0, 100]
        
        # Filter: keep lines that span full TAB height
        filtered = [0]  # Start
        for x in measure_xs:
            if filtered and x - filtered[-1] > 5:  # At least 5% apart
                filtered.append(x)
        filtered.append(100)  # End
        
        return filtered
    
    def analyze(self, image_path: str, debug: bool = False) -> Dict:
        """
        Full analysis:
        1. Detect TAB lines
        2. Analyze each line
        3. Group by X position to form beats
        4. Split by measure boundaries
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        h, w = image.shape[:2]
        
        # For full page, we need to process each system
        # For now, assume single system or use first detected
        
        print("[LineByLine] Detecting TAB lines...")
        line_ys = self.detect_tab_lines(image)
        print(f"[LineByLine] Lines at Y: {line_ys}")
        
        # Analyze each line
        all_notes: List[NotePosition] = []
        
        for string_num, y in enumerate(line_ys, 1):
            print(f"[LineByLine] Analyzing string {string_num}...")
            strip = self.extract_line_strip(image, y, strip_height=30)
            
            if debug:
                cv2.imwrite(f"debug_line_{string_num}.png", strip)
            
            frets = self.analyze_line(strip, string_num)
            
            for f in frets:
                fret_val = f.get("fret")
                x_pct = f.get("x_percent", 0)
                if fret_val is not None:
                    all_notes.append(NotePosition(
                        string=string_num,
                        fret=fret_val,
                        x_pos=x_pct
                    ))
            
            print(f"[LineByLine] String {string_num}: {len(frets)} notes")
        
        print(f"[LineByLine] Total notes: {len(all_notes)}")
        
        # Group into beats
        beats = self.group_into_beats(all_notes)
        print(f"[LineByLine] Grouped into {len(beats)} beats")
        
        # Detect measures and split
        measure_xs = self.detect_measure_boundaries(image, line_ys)
        print(f"[LineByLine] Measure boundaries: {measure_xs}")
        
        # Split beats into measures
        measures = []
        current_measure_beats = []
        measure_idx = 0
        
        for beat in beats:
            # Estimate beat X position from notes
            if beat["notes"]:
                avg_x = sum(n.x_pos for n in all_notes 
                           if any(n.string == bn["string"] and n.fret == bn["fret"] 
                                 for bn in beat["notes"])) / len(beat["notes"])
            else:
                avg_x = 0
            
            # Check if we crossed a measure boundary
            if measure_idx + 1 < len(measure_xs) and avg_x > measure_xs[measure_idx + 1]:
                if current_measure_beats:
                    measures.append({"number": len(measures) + 1, 
                                   "beats": current_measure_beats})
                current_measure_beats = []
                measure_idx += 1
            
            current_measure_beats.append(beat)
        
        if current_measure_beats:
            measures.append({"number": len(measures) + 1, 
                           "beats": current_measure_beats})
        
        return {
            "measures": measures,
            "total_notes": len(all_notes),
            "total_beats": len(beats)
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m omnitab.tab_ocr.line_by_line_analyzer <image>")
        sys.exit(1)
    
    analyzer = LineByLineAnalyzer()
    result = analyzer.analyze(sys.argv[1], debug=True)
    
    print("\n=== Result ===")
    print(f"Measures: {len(result['measures'])}")
    print(f"Notes: {result['total_notes']}")
    print(f"Beats: {result['total_beats']}")
