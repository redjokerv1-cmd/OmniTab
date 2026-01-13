"""Test Gemini + OCR complete pipeline"""

from omnitab.tab_ocr.complete_converter import CompleteConverter

print('=== Gemini + OCR Complete Test ===')
print()

converter = CompleteConverter()
result = converter.convert(
    image_path='test_samples/images/page_1.png',
    output_path='test_samples/output/page_1_gemini.gp5',
    title='Yellow Jacket - Gemini',
    use_gemini=True
)

print()
print('=' * 50)
print('FINAL RESULT')
print('=' * 50)
print(f"  Rhythm source: {result['rhythm_source']}")
print(f"  Measures: {result['measures']}")
print(f"  Notes: {result['notes']}")
print(f"  Tuning: {result['tuning']}")
print(f"  Capo: {result['capo']}")
