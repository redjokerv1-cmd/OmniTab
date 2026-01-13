"""Optimize line removal parameters"""
import cv2
from omnitab.tab_ocr.recognizer.simple_binary_ocr import SimpleBinaryOCR
from omnitab.tab_ocr.recognizer.line_remover import StaffLineRemover

image = cv2.imread('test_samples/images/page_1.png')

print("Testing different kernel lengths...")
print("="*70)
print(f"{'kernel':>6} | {'repair':>6} | {'digits':>6} | {'chords':>6} | {'problems':>8} | {'avg_conf':>8}")
print("-"*70)

# Test different parameters
for kernel_len in [40, 60, 80, 100, 120]:
    for repair in [0, 1, 2, 3]:
        remover = StaffLineRemover(kernel_length=kernel_len, repair_kernel=repair)
        no_lines, _ = remover.remove_lines(image)
        
        no_lines_bgr = cv2.cvtColor(no_lines, cv2.COLOR_GRAY2BGR)
        ocr = SimpleBinaryOCR(min_confidence=0.2)
        result = ocr.process(no_lines_bgr)
        
        problems = sum(1 for s in result['systems'] for c in s.chords if len(c.notes) > 6)
        digits = len(result['digits'])
        chords = result['total_chords']
        conf = result['avg_confidence'] * 100
        
        status = "GOOD" if problems == 0 and digits > 100 else ""
        print(f"{kernel_len:>6} | {repair:>6} | {digits:>6} | {chords:>6} | {problems:>8} | {conf:>7.1f}% {status}")

print("-"*70)

# Best case: no problems + most digits
print("\nLooking for: problems=0 AND digits > 100")
