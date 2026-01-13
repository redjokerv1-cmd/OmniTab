"""Test OCR after line removal"""
import cv2
from omnitab.tab_ocr.recognizer.simple_binary_ocr import SimpleBinaryOCR
from omnitab.tab_ocr.recognizer.line_remover import StaffLineRemover

# Load image
image = cv2.imread('test_samples/images/page_1.png')

# Step 1: Remove lines
print("Step 1: Removing staff lines...")
remover = StaffLineRemover()
no_lines, lines_mask = remover.remove_lines(image)

# Save intermediate result
cv2.imwrite('test_samples/output/step1_no_lines.png', no_lines)
cv2.imwrite('test_samples/output/step1_lines_mask.png', lines_mask)

# Step 2: OCR on clean image
print("Step 2: Running OCR...")
no_lines_bgr = cv2.cvtColor(no_lines, cv2.COLOR_GRAY2BGR)
ocr = SimpleBinaryOCR(min_confidence=0.2)
result = ocr.process(no_lines_bgr)

print()
print("="*60)
print("AFTER LINE REMOVAL")
print("="*60)
print(f"Recognized: {len(result['digits'])} digits")
print(f"Systems: {len(result['systems'])}")
print(f"Chords: {result['total_chords']}")
print(f"Avg confidence: {result['avg_confidence']*100:.1f}%")

# Draw boxes on original image
original = image.copy()
for digit in result['digits']:
    x = int(digit.x - digit.width/2)
    y = int(digit.y - digit.height/2)
    w = digit.width
    h = digit.height
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(original, str(digit.value), (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite('test_samples/output/ocr_after_line_removal.png', original)
print()
print("Saved: test_samples/output/ocr_after_line_removal.png")

# Analyze chord quality
print()
print("="*60)
print("CHORD ANALYSIS (should be max 6 notes per chord)")
print("="*60)
problems = 0
for i, system in enumerate(result['systems']):
    print(f"\nSystem {i+1}: {len(system.chords)} chords")
    for j, chord in enumerate(system.chords):
        note_count = len(chord.notes)
        frets = chord.values
        conf = chord.avg_confidence
        
        status = "OK" if note_count <= 6 else "PROBLEM"
        if note_count > 6:
            problems += 1
        
        if j < 5 or note_count > 6:
            print(f"  Chord {j+1}: {note_count} notes, frets={frets[:6]}{'...' if len(frets)>6 else ''} [{status}]")
    
    if len(system.chords) > 5:
        remaining = len(system.chords) - 5
        ok_count = sum(1 for c in system.chords[5:] if len(c.notes) <= 6)
        print(f"  ... +{remaining} more ({ok_count} OK)")

print()
print(f"Total problems (chords with >6 notes): {problems}")
