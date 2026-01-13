"""
Full Test Suite - Test on all available images
"""

import os
from omnitab.tab_ocr.smart_to_gp5 import SmartToGp5
from omnitab.learning.db import LearningDB

def run_tests():
    print("=" * 70)
    print("FULL TEST SUITE")
    print("=" * 70)
    
    # Find all test images
    image_dir = "test_samples/images"
    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    print(f"Found {len(images)} test images: {images}")
    
    converter = SmartToGp5()
    results = []
    
    for img_name in sorted(images):
        img_path = os.path.join(image_dir, img_name)
        out_name = img_name.replace(".png", ".gp5")
        out_path = os.path.join("test_samples/output", out_name)
        
        print(f"\n--- Processing: {img_name} ---")
        
        try:
            result = converter.convert(img_path, out_path, img_name.replace(".png", ""), 65)
            results.append({
                "image": img_name,
                "notes": result["gp5_notes"],
                "digits": result["merged_digits"],
                "two_digit": result["two_digit_frets"],
                "chords": result["chords_valid"]
            })
            print(f"SUCCESS: {result['gp5_notes']} notes")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"image": img_name, "notes": 0, "error": str(e)})
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    total_notes = 0
    for r in results:
        notes = r.get("notes", 0)
        total_notes += notes
        two_d = r.get("two_digit", 0)
        chords = r.get("chords", 0)
        print(f"{r['image']:20} | {notes:3} notes | {two_d:2} 2-digit | {chords:2} chords")
    
    print(f"\nTotal GP5 notes across all pages: {total_notes}")
    
    # DB stats
    db = LearningDB()
    stats = db.get_stats()
    print(f"\nLearning DB:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Avg mapping rate: {stats['avg_mapping_rate']*100:.1f}%")
    
    return results


if __name__ == "__main__":
    run_tests()
