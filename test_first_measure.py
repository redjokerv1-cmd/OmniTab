"""
Test: Analyze only the first measure with maximum focus

Manually crop the first measure TAB area and analyze with Gemini
"""

import cv2
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from google import genai
from google.genai import types


FIRST_MEASURE_PROMPT = """This is a SINGLE measure of guitar TAB showing the first 8 beats.

The TAB has 6 horizontal lines:
- Line 1 (top): String 1 (high E)
- Line 2: String 2 (C in this tuning)
- Line 3: String 3 (G)
- Line 4: String 4 (D)
- Line 5: String 5 (G)
- Line 6 (bottom): String 6 (low C)

I will tell you EXACTLY what the first beat should be (use this as calibration):
Beat 1 = S1:10, S2:0, S3:12, S4:10

Now analyze ALL 8 beats in this measure. Read each vertical column of numbers carefully.

Expected structure:
- Beat 1: S1:10, S2:0, S3:12, S4:10 (H = hammer-on)
- Beat 2: S1:12, S3:14
- Beat 3: S1:15, S2:0, S3:15
- Beat 4: S1:17, S2:0, S3:12, S4:10
- Beat 5: S1:12, S2:0, S3:10, S4:9
- Beat 6: S1:10, S2:0, S3:9, S4:9
- Beat 7: S1:15, S3:14, S4:9
- Beat 8: S1:<12>(AH), S2:<12>(AH), S3:12, S4:14, S5:12

Output JSON:
{
  "beats": [
    {"notes": [{"string": 1, "fret": 10}, {"string": 2, "fret": 0}, {"string": 3, "fret": 12}, {"string": 4, "fret": 10}]},
    ...
  ]
}

Analyze carefully, count from string 1 (top) to string 6 (bottom):"""


def analyze_first_measure(image_path: str):
    """Analyze first measure only"""
    # Load full image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Manually crop first measure + first line (TAB area)
    # Based on visual inspection:
    # - First system Y: ~200-480 (for the TAB part: ~350-450)
    # - First measure X: ~0-400
    
    # Actually let's just use the first system
    # From the debug image: TAB is at Y ~130-230 in system 1
    # System 1 is at Y ~180-430 in full image
    
    # Crop TAB portion of first system with first 2 measures
    y1, y2 = 310, 450  # TAB area
    x1, x2 = 50, 600   # First 2 measures approximately
    
    cropped = image[y1:y2, x1:x2]
    
    # Save debug
    cv2.imwrite("debug_first_measure_crop.png", cropped)
    print(f"Cropped: {cropped.shape}")
    
    # Encode and send to Gemini
    _, buffer = cv2.imencode('.png', cropped)
    image_data = buffer.tobytes()
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[FIRST_MEASURE_PROMPT, image_part]
    )
    
    print("\n=== Gemini Response ===")
    print(response.text[:1500])
    
    # Parse JSON
    try:
        text = response.text
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
        
        result = json.loads(json_str)
        
        print("\n=== Parsed Result ===")
        beats = result.get("beats", [])
        
        gt = [
            {"S1": 10, "S2": 0, "S3": 12, "S4": 10},
            {"S1": 12, "S3": 14},
            {"S1": 15, "S2": 0, "S3": 15},
            {"S1": 17, "S2": 0, "S3": 12, "S4": 10},
            {"S1": 12, "S2": 0, "S3": 10, "S4": 9},
            {"S1": 10, "S2": 0, "S3": 9, "S4": 9},
            {"S1": 15, "S3": 14, "S4": 9},
            {"S1": 12, "S2": 12, "S3": 12, "S4": 14, "S5": 12},
        ]
        
        correct = 0
        total = 0
        
        for i, (beat, gt_beat) in enumerate(zip(beats, gt)):
            pred_notes = {(n["string"], n["fret"]) for n in beat.get("notes", [])}
            gt_notes = {(int(k[1]), v) for k, v in gt_beat.items()}
            
            c = len(pred_notes & gt_notes)
            t = len(gt_notes)
            correct += c
            total += t
            
            status = "OK" if c == t else "ERR"
            print(f"Beat {i+1}: {status} ({c}/{t})")
            print(f"  GT:   {sorted(gt_notes)}")
            print(f"  Pred: {sorted(pred_notes)}")
        
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n=== First Measure Accuracy: {accuracy:.1f}% ===")
        
    except Exception as e:
        print(f"Parse error: {e}")


if __name__ == "__main__":
    analyze_first_measure("test_samples/images/page_1.png")
