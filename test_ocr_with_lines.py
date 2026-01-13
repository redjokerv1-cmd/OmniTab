"""Test OCR with accurate line detection"""
import cv2
from omnitab.tab_ocr.recognizer.enhanced_ocr import EnhancedTabOCR
from omnitab.tab_ocr.recognizer.horizontal_projection import HorizontalProjection

# Load image
image = cv2.imread('test_samples/images/page_1.png')

# Step 1: Detect TAB lines (수평 투영)
print("Step 1: Detecting TAB lines...")
line_detector = HorizontalProjection()
systems = line_detector.detect(image)

print(f"Found {len(systems)} TAB systems")
for i, system in enumerate(systems):
    print(f"  System {i+1}: Y={system.y_start:.0f}-{system.y_end:.0f}, confidence={system.confidence:.1%}")

# Step 2: Run OCR
print("\nStep 2: Running OCR...")
ocr = EnhancedTabOCR()
result = ocr.process(image)

digits = result['digits']
print(f"Found {len(digits)} digits")

# Step 3: Map digits to strings using detected lines
print("\nStep 3: Mapping digits to strings...")

mapped_digits = []
unmapped_count = 0

for digit in digits:
    # Find which system this digit belongs to
    matched_system = None
    for system in systems:
        if system.y_start <= digit.y <= system.y_end:
            matched_system = system
            break
    
    if matched_system:
        string_num = matched_system.get_string_for_y(digit.y)
        if string_num > 0:
            mapped_digits.append({
                'value': digit.value,
                'string': string_num,
                'x': digit.x,
                'y': digit.y,
                'confidence': digit.confidence
            })
        else:
            unmapped_count += 1
    else:
        unmapped_count += 1

print(f"Mapped: {len(mapped_digits)} digits")
print(f"Unmapped: {unmapped_count} digits")

# Step 4: Show sample mappings
print("\nSample mappings (first 20):")
print("-" * 50)
for d in sorted(mapped_digits, key=lambda x: x['x'])[:20]:
    print(f"  Fret {d['value']:2d} on String {d['string']} at X={d['x']:.0f}")

# Step 5: Group into chords and show
print("\nChord grouping with string numbers:")
print("-" * 50)

# Group by X position
x_sorted = sorted(mapped_digits, key=lambda x: x['x'])
chords = []
current_chord = [x_sorted[0]] if x_sorted else []

for d in x_sorted[1:]:
    if d['x'] - current_chord[-1]['x'] < 15:  # Same chord
        current_chord.append(d)
    else:
        chords.append(current_chord)
        current_chord = [d]
if current_chord:
    chords.append(current_chord)

print(f"Found {len(chords)} chords")
print()

# Show first 10 chords
for i, chord in enumerate(chords[:10]):
    notes = sorted(chord, key=lambda x: x['string'])
    chord_str = ", ".join([f"S{n['string']}:{n['value']}" for n in notes])
    print(f"  Chord {i+1}: [{chord_str}]")

# Accuracy check
print("\n" + "=" * 60)
print("ACCURACY CHECK")
print("=" * 60)

# Count notes per string
string_counts = {i: 0 for i in range(1, 7)}
for d in mapped_digits:
    string_counts[d['string']] += 1

print("Notes per string:")
for s in range(1, 7):
    bar = "█" * (string_counts[s] // 2)
    print(f"  String {s}: {string_counts[s]:3d} {bar}")

# Valid chord check (max 6 notes)
valid_chords = sum(1 for c in chords if len(c) <= 6)
print(f"\nValid chords (≤6 notes): {valid_chords}/{len(chords)}")
