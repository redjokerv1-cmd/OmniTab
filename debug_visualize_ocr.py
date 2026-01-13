"""
Debug Visualizer - Overlay OCR results on original image
to visually verify accuracy
"""

import cv2
import numpy as np
from pathlib import Path
from omnitab.tab_ocr.recognizer.enhanced_ocr import EnhancedTabOCR
from omnitab.tab_ocr.recognizer.horizontal_projection import HorizontalProjection


def visualize_ocr(image_path: str, output_path: str = None):
    """Overlay OCR results on original image"""
    
    print(f"Loading: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load {image_path}")
    
    output = image.copy()
    height, width = image.shape[:2]
    
    # 1. Detect TAB lines
    print("Detecting TAB lines...")
    line_detector = HorizontalProjection()
    systems = line_detector.detect(image)
    print(f"  Found {len(systems)} systems")
    
    # Draw TAB lines (blue)
    for sys_idx, system in enumerate(systems):
        for i, y in enumerate(system.line_y_positions):
            cv2.line(output, (0, int(y)), (width, int(y)), (255, 0, 0), 1)
            # Label string number
            cv2.putText(output, f"S{i+1}", (5, int(y) - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # 2. Run OCR
    print("Running OCR...")
    ocr = EnhancedTabOCR(use_gpu=False)
    result = ocr.process(image)
    digits = result['digits']
    print(f"  Found {len(digits)} digits")
    
    # 3. Map digits to strings
    mapped = []
    unmapped = []
    
    for digit in digits:
        found = False
        for system in systems:
            if system.y_start <= digit.y <= system.y_end:
                string_num = system.get_string_for_y(digit.y)
                if string_num > 0:
                    mapped.append((digit, string_num))
                    found = True
                break
        if not found:
            unmapped.append(digit)
    
    print(f"  Mapped: {len(mapped)}, Unmapped: {len(unmapped)}")
    
    # 4. Draw OCR results
    # Green = correctly mapped, Red = unmapped, Yellow = suspicious (fret > 19)
    
    for digit, string_num in mapped:
        x, y = int(digit.x), int(digit.y)
        fret = int(digit.value)
        
        # Color based on fret value
        if fret > 19:  # Suspicious - likely OCR error
            color = (0, 255, 255)  # Yellow
            label = f"{fret}?"
        else:
            color = (0, 255, 0)  # Green
            label = str(fret)
        
        # Draw rectangle around digit
        cv2.rectangle(output, (x - 8, y - 8), (x + 8, y + 8), color, 1)
        # Draw fret number
        cv2.putText(output, label, (x - 5, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        # Draw string assignment
        cv2.putText(output, f"S{string_num}", (x - 5, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
    
    # Draw unmapped (red)
    for digit in unmapped:
        x, y = int(digit.x), int(digit.y)
        fret = int(digit.value)
        cv2.rectangle(output, (x - 8, y - 8), (x + 8, y + 8), (0, 0, 255), 2)
        cv2.putText(output, str(fret), (x - 5, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # 5. Statistics overlay
    stats = [
        f"Systems: {len(systems)}",
        f"Digits: {len(digits)}",
        f"Mapped: {len(mapped)}",
        f"Unmapped: {len(unmapped)}",
        f"Suspicious (>19): {sum(1 for d, _ in mapped if d.value > 19)}"
    ]
    
    for i, stat in enumerate(stats):
        cv2.putText(output, stat, (10, 20 + i * 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(output, stat, (10, 20 + i * 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 6. Save
    if output_path is None:
        output_path = str(Path(image_path).stem) + "_debug.png"
    
    cv2.imwrite(output_path, output)
    print(f"\nSaved: {output_path}")
    
    # 7. Analyze errors
    print("\n=== Error Analysis ===")
    suspicious = [(d, s) for d, s in mapped if d.value > 19]
    if suspicious:
        print(f"Suspicious frets (likely OCR errors):")
        for digit, string in suspicious:
            print(f"  Fret {int(digit.value)} at ({int(digit.x)}, {int(digit.y)}) -> String {string}")
    
    # Fret distribution
    frets = [int(d.value) for d, _ in mapped]
    from collections import Counter
    print(f"\nFret distribution (top 10):")
    for fret, count in Counter(frets).most_common(10):
        print(f"  Fret {fret:2d}: {count} times")
    
    return {
        'systems': len(systems),
        'digits': len(digits),
        'mapped': len(mapped),
        'unmapped': len(unmapped),
        'suspicious': len(suspicious),
        'output_path': output_path
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        image_path = "test_samples/images/page_1.png"
    else:
        image_path = sys.argv[1]
    
    output_path = "test_samples/output/ocr_debug.png"
    
    result = visualize_ocr(image_path, output_path)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Accuracy estimate: {result['mapped'] / result['digits'] * 100:.1f}% mapped")
    print(f"Error rate: {result['suspicious'] / result['mapped'] * 100:.1f}% suspicious")
