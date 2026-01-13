"""Full conversion test with correct Yellow Jacket tuning"""
from omnitab.tab_ocr.ocr_to_gp5 import convert_tab_to_gp5

# Yellow Jacket correct tuning:
# ①=E ②=C ③=G ④=D ⑤=G ⑥=C
YELLOW_JACKET_TUNING = ['E', 'C', 'G', 'D', 'G', 'C']

result = convert_tab_to_gp5(
    image_path="test_samples/images/page_1.png",
    output_path="test_samples/output/yellow_jacket_correct.gp5",
    title="Yellow Jacket - Shaun Martin",
    tempo=65,
    tuning=YELLOW_JACKET_TUNING,
    capo=2
)

print()
print("="*60)
print("FINAL RESULT")
print("="*60)
print(f"File: {result['output_path']}")
print(f"Beats: {result['total_beats']}")
print(f"Tuning: {result['tuning']}")
print(f"Capo: {result['capo']}")
print()
print("Open in Guitar Pro to verify!")
