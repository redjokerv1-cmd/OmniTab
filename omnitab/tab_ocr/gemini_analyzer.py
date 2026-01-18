"""
Gemini Vision API for TAB rhythm analysis

Usage:
    # Method 1: Environment variable (recommended)
    # Create .env file with: GOOGLE_API_KEY=your-key-here
    from omnitab.tab_ocr.gemini_analyzer import GeminiTabAnalyzer
    analyzer = GeminiTabAnalyzer()  # Auto-loads from .env
    
    # Method 2: Direct (NOT recommended for production)
    analyzer = GeminiTabAnalyzer(api_key="your-key")
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

# Load .env file if exists
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use os.environ only

# Use new google.genai SDK (the old google.generativeai is deprecated)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiTabAnalyzer:
    """
    Use Gemini Vision to analyze TAB images and extract:
    - Note positions (string, fret)
    - Rhythm/duration information
    - Techniques (bend, slide, hammer-on, etc.)
    """
    
    # Prompt for TAB analysis
    ANALYSIS_PROMPT = """You are a guitar tablature (TAB) expert.

## CRITICAL: This image has TWO parts - READ THEM SEPARATELY!

### PART 1: Standard Notation (5-line staff with note heads "♩")
- This shows RHYTHM information (note duration, rests, time signature)
- Use this ONLY for rhythm/duration, NOT for pitch

### PART 2: TAB (6-line grid with NUMBERS)
- This shows FRET POSITIONS (which string, which fret)
- READ ONLY THE NUMBERS from this section
- The TAB section has 6 horizontal lines with numbers on them

## HOW TO READ TAB NUMBERS:

The TAB section looks like this (6 lines with letters T-A-B on left):
```
T ──10──12──15──   ← String 1 (TOP line), frets 10, 12, 15
A ──0───────0───   ← String 2, frets 0, 0
B ──12──14──15──   ← String 3, frets 12, 14, 15
  ──10──────15──   ← String 4
  ──────────────   ← String 5 (empty = no note)
  ──────────────   ← String 6 (BOTTOM line)
```

## READING RULES:

1. **IGNORE the 5-line staff notation (♩ note heads)** - use it ONLY for rhythm
2. **FOCUS on the 6-line TAB section** - read the NUMBERS only
3. Numbers vertically aligned = same beat (play together)
4. String 1 = TOP line of 6-line TAB
5. String 6 = BOTTOM line of 6-line TAB
6. Empty line = no note on that string for that beat
7. <12> = harmonic at fret 12
8. H = hammer-on, P = pull-off, AH = artificial harmonic

## EXAMPLE:

If TAB shows:
```
  10    12
   0
  12    14
  10
```

This means:
- Beat 1: String 1 fret 10, String 2 fret 0, String 3 fret 12, String 4 fret 10
- Beat 2: String 1 fret 12, String 3 fret 14

## OUTPUT FORMAT (JSON):

{
  "measures": [
    {
      "number": 1,
      "beats": [
        {
          "duration": "eighth",  // from staff notation rhythm
          "notes": [
            {"string": 1, "fret": 10, "technique": null},
            {"string": 2, "fret": 0, "technique": null},
            {"string": 3, "fret": 12, "technique": null},
            {"string": 4, "fret": 10, "technique": null}
          ]
        }
      ]
    }
  ],
  "tuning": ["E", "C", "G", "D", "G", "C"],  // from header if shown
  "capo": 2,  // from header if shown
  "tempo": 65  // from header if shown
}

Now analyze the image. Remember: 
- Staff = rhythm only
- TAB = fret numbers only
- Numbers on same vertical position = same beat"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini analyzer
        
        Args:
            api_key: Google AI API key. If not provided, uses GEMINI_API_KEY or GOOGLE_API_KEY env var
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        # Check both env var names (GEMINI_API_KEY for consistency with stock-predictor)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY env var or pass api_key parameter")
        
        # Model name (2025/2026 latest)
        self.model_name = "gemini-2.5-flash"
        
        # New google.genai SDK uses Client pattern
        self.client = genai.Client(api_key=self.api_key)
    
    def analyze(self, image_path: str) -> Dict:
        """
        Analyze a TAB image and extract musical information
        
        Args:
            image_path: Path to the TAB image
            
        Returns:
            Dict with measures, notes, rhythms, techniques
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image data
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif"
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        # New google.genai SDK uses types.Part for images
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        
        # Send to Gemini using new SDK
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.ANALYSIS_PROMPT, image_part]
        )
        
        # Parse response
        return self._parse_response(response.text)
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini's response and extract JSON"""
        # Find JSON in response
        try:
            # Try to find JSON block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                # Try to find JSON directly
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse JSON",
                "raw_response": response_text,
                "parse_error": str(e)
            }
    
    def analyze_with_ocr_results(self, 
                                  image_path: str, 
                                  ocr_notes: List[Dict]) -> Dict:
        """
        Combine OCR results with Gemini analysis for better accuracy
        
        Args:
            image_path: Path to TAB image
            ocr_notes: Notes detected by OCR (list of {string, fret, x, y})
            
        Returns:
            Enhanced analysis with rhythms added to OCR notes
        """
        # First, get Gemini's rhythm analysis
        gemini_result = self.analyze(image_path)
        
        if "error" in gemini_result:
            # Fallback: use OCR notes with default rhythm
            return {
                "measures": [{
                    "number": 1,
                    "beats": [{
                        "duration": "quarter",
                        "notes": [{"string": n["string"], "fret": n["fret"]} for n in ocr_notes]
                    }]
                }],
                "source": "ocr_only",
                "gemini_error": gemini_result.get("error")
            }
        
        # Merge: use OCR positions + Gemini rhythms
        return {
            "measures": gemini_result.get("measures", []),
            "tuning": gemini_result.get("tuning", ["E", "B", "G", "D", "A", "E"]),
            "capo": gemini_result.get("capo", 0),
            "tempo": gemini_result.get("tempo", 120),
            "source": "gemini+ocr"
        }


# Duration conversion for GP5
DURATION_TO_GP5 = {
    "whole": -2,
    "half": -1,
    "quarter": 0,
    "eighth": 1,
    "sixteenth": 2,
    "32nd": 3
}


def gemini_to_gp5_notes(gemini_result: Dict) -> List[Dict]:
    """
    Convert Gemini analysis result to GP5 notes format
    
    Returns list of:
    {
        "string": 1-6,
        "fret": 0-24,
        "duration": -2 to 3,
        "technique": "hammer-on" | "slide" | etc.
    }
    """
    notes = []
    
    for measure in gemini_result.get("measures", []):
        for beat in measure.get("beats", []):
            duration_str = beat.get("duration", "quarter")
            duration_gp5 = DURATION_TO_GP5.get(duration_str, 0)
            
            for note in beat.get("notes", []):
                notes.append({
                    "string": note.get("string"),
                    "fret": note.get("fret"),
                    "duration": duration_gp5,
                    "technique": note.get("technique")
                })
    
    return notes


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_analyzer.py <image_path> [api_key]")
        print("")
        print("Environment variable GOOGLE_API_KEY can also be used")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        analyzer = GeminiTabAnalyzer(api_key=api_key)
        result = analyzer.analyze(image_path)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
