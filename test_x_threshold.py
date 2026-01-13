"""Test different X thresholds for chord grouping"""
import cv2
from omnitab.tab_ocr.recognizer.simple_binary_ocr import SimpleBinaryOCR

image = cv2.imread('test_samples/images/page_1.png')

print("Testing X threshold (without line removal)...")
print("="*70)
print(f"{'x_thresh':>8} | {'digits':>6} | {'chords':>6} | {'problems':>8} | {'max_notes':>10}")
print("-"*70)

for x_thresh in [5, 10, 15, 20, 25, 30]:
    ocr = SimpleBinaryOCR(min_confidence=0.2)
    result = ocr.process(image)
    
    # Re-group with different threshold
    from omnitab.tab_ocr.recognizer.simple_binary_ocr import TabChordData
    
    for system in result['systems']:
        digits = sorted(system.digits, key=lambda d: d.x)
        chords = []
        if digits:
            current = [digits[0]]
            for d in digits[1:]:
                if d.x - current[-1].x < x_thresh:
                    current.append(d)
                else:
                    chords.append(TabChordData(current, sum(n.x for n in current)/len(current)))
                    current = [d]
            chords.append(TabChordData(current, sum(n.x for n in current)/len(current)))
        system.chords = chords
    
    total_chords = sum(len(s.chords) for s in result['systems'])
    problems = sum(1 for s in result['systems'] for c in s.chords if len(c.notes) > 6)
    max_notes = max((len(c.notes) for s in result['systems'] for c in s.chords), default=0)
    
    status = "GOOD" if problems == 0 else ""
    print(f"{x_thresh:>8} | {len(result['digits']):>6} | {total_chords:>6} | {problems:>8} | {max_notes:>10} {status}")

print("-"*70)
