"""
Deep Learning Cycles - Continuous improvement until results are satisfactory
"""

import cv2
import os
from datetime import datetime

def run_all_cycles():
    print("=" * 70)
    print("DEEP LEARNING CYCLES - CONTINUOUS RUN")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # ============ CYCLE 3: Fix tuning issue ============
    print("\n" + "=" * 70)
    print("CYCLE 3: Tuning Detection Analysis")
    print("=" * 70)
    
    import easyocr
    from omnitab.tab_ocr.recognizer.header_detector import HeaderDetector
    
    image = cv2.imread("test_samples/images/page_1.png")
    height = image.shape[0]
    header_region = image[0:int(height * 0.25), :]
    
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(header_region, detail=1)
    
    print("Header OCR results:")
    for r in results:
        text = r[1]
        conf = r[2]
        if conf > 0.3:
            print(f"  '{text}' (conf={conf:.2f})")
    
    full_text = " ".join([r[1] for r in results])
    print(f"\nFull text: {full_text}")
    
    # ============ CYCLE 4: Test on page 2 ============
    print("\n" + "=" * 70)
    print("CYCLE 4: Testing on Different Page")
    print("=" * 70)
    
    from omnitab.tab_ocr.smart_to_gp5 import SmartToGp5
    
    page2_path = "test_samples/images/page_2.png"
    if os.path.exists(page2_path):
        print(f"Processing {page2_path}...")
        converter = SmartToGp5()
        result = converter.convert(
            page2_path,
            "test_samples/output/yellow_jacket_page2.gp5",
            "Yellow Jacket Page 2",
            65
        )
        print(f"\nPage 2 Result: {result['gp5_notes']} notes")
    else:
        print(f"Page 2 not found at {page2_path}")
    
    # ============ CYCLE 5: Compare all attempts ============
    print("\n" + "=" * 70)
    print("CYCLE 5: Compare All Attempts")
    print("=" * 70)
    
    from omnitab.learning.db import LearningDB
    
    db = LearningDB()
    attempts = db.get_all_attempts(limit=10)
    
    print(f"Total attempts in DB: {len(attempts)}")
    print("\nAttempt History:")
    for a in sorted(attempts, key=lambda x: x.timestamp):
        method = a.settings.get("method", "original")
        rate = a.mapped_digits / max(a.total_digits, 1) * 100
        print(f"  {a.timestamp.strftime('%H:%M:%S')} | {method:12} | {a.gp5_notes:3} notes | {rate:.1f}% mapping")
    
    # Best result
    if attempts:
        best = max(attempts, key=lambda x: x.gp5_notes)
        print(f"\nBest Result:")
        print(f"  Method: {best.settings.get('method', 'unknown')}")
        print(f"  GP5 Notes: {best.gp5_notes}")
        print(f"  Mapping: {best.mapped_digits}/{best.total_digits}")
    
    # ============ CYCLE 6: Final optimization ============
    print("\n" + "=" * 70)
    print("CYCLE 6: Final Optimization Test")
    print("=" * 70)
    
    converter = SmartToGp5()
    result = converter.convert(
        "test_samples/images/page_1.png",
        "test_samples/output/yellow_jacket_final.gp5",
        "Yellow Jacket",
        65
    )
    
    # ============ CYCLE 7: Improvement metrics ============
    print("\n" + "=" * 70)
    print("CYCLE 7: Calculate Improvement")
    print("=" * 70)
    
    # Get all attempts again
    attempts = db.get_all_attempts(limit=20)
    
    # Original method attempts
    original = [a for a in attempts if a.settings.get("method") != "smart_ocr"]
    smart = [a for a in attempts if a.settings.get("method") == "smart_ocr"]
    
    if original and smart:
        orig_best = max(a.gp5_notes for a in original)
        smart_best = max(a.gp5_notes for a in smart)
        improvement = ((smart_best - orig_best) / orig_best) * 100
        
        print(f"Original method best: {orig_best} notes")
        print(f"Smart OCR best: {smart_best} notes")
        print(f"Improvement: {improvement:.1f}%")
    
    # ============ FINAL REPORT ============
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"GP5 Notes: {result['gp5_notes']}")
    print(f"2-digit frets: {result['two_digit_frets']}")
    print(f"Valid chords: {result['chords_valid']}/{result['chords_total']}")
    print(f"Tuning: {result['tuning']}")
    print(f"Capo: {result['capo']}")
    
    print("\nCompleted:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_all_cycles()
