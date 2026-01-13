"""Hybrid approach: line removal + small x_threshold"""
import cv2
from omnitab.tab_ocr.recognizer.simple_binary_ocr import SimpleBinaryOCR, TabChordData
from omnitab.tab_ocr.recognizer.line_remover import StaffLineRemover

image = cv2.imread('test_samples/images/page_1.png')

print("Testing HYBRID (line removal + x_thresh)...")
print("="*80)
print(f"{'kernel':>6} | {'repair':>6} | {'x_th':>4} | {'digits':>6} | {'chords':>6} | {'problems':>8} | {'max':>4}")
print("-"*80)

best = None

for kernel in [40, 60]:
    for repair in [1, 2]:
        # Step 1: Remove lines
        remover = StaffLineRemover(kernel_length=kernel, repair_kernel=repair)
        no_lines, _ = remover.remove_lines(image)
        no_lines_bgr = cv2.cvtColor(no_lines, cv2.COLOR_GRAY2BGR)
        
        for x_thresh in [5, 8, 10]:
            # Step 2: OCR
            ocr = SimpleBinaryOCR(min_confidence=0.2)
            result = ocr.process(no_lines_bgr)
            
            # Step 3: Re-group with x_thresh
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
            num_digits = len(result['digits'])
            
            status = "BEST" if problems == 0 and num_digits > 100 else ("GOOD" if problems == 0 else "")
            print(f"{kernel:>6} | {repair:>6} | {x_thresh:>4} | {num_digits:>6} | {total_chords:>6} | {problems:>8} | {max_notes:>4} {status}")
            
            if problems == 0 and (best is None or num_digits > best[0]):
                best = (num_digits, kernel, repair, x_thresh, total_chords)

print("-"*80)

if best:
    print(f"\nBEST: {best[0]} digits with kernel={best[1]}, repair={best[2]}, x_thresh={best[3]}")
    print(f"      {best[4]} chords, 0 problems")
else:
    print("\nNo perfect solution found. Using best compromise...")
