"""Debug: Visualize OCR results"""
import cv2
from omnitab.tab_ocr.recognizer.simple_binary_ocr import SimpleBinaryOCR

# Load image
image = cv2.imread('test_samples/images/page_1.png')
original = image.copy()

# Run OCR
ocr = SimpleBinaryOCR(min_confidence=0.2)
result = ocr.process(image)

digits = result['digits']
systems = result['systems']
print(f"Recognized: {len(digits)} digits")
print(f"Systems: {len(systems)}")
print(f"Chords: {result['total_chords']}")
print(f"Avg confidence: {result['avg_confidence']:.2%}")

# Draw green boxes on recognized digits
for digit in digits:
    x = int(digit.x - digit.width/2)
    y = int(digit.y - digit.height/2)
    w = digit.width
    h = digit.height
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(original, str(digit.value), (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Save
cv2.imwrite('test_samples/output/ocr_debug_result.png', original)
print("\nSaved: test_samples/output/ocr_debug_result.png")

# System details
print("\n" + "="*60)
for i, system in enumerate(systems):
    print(f"\nSystem {i+1}: Y={system.y_start:.0f}-{system.y_end:.0f}")
    print(f"  Digits: {len(system.digits)}")
    print(f"  Chords: {len(system.chords)}")
    
    for j, chord in enumerate(system.chords[:5]):
        frets = chord.values
        conf = chord.avg_confidence
        print(f"    Chord {j+1}: frets={frets} conf={conf:.0%}")
    
    if len(system.chords) > 5:
        print(f"    ... +{len(system.chords)-5} more")
