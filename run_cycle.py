"""
Learning Cycle Runner - Iterative improvement
"""

import cv2
from omnitab.learning.analyzer import LearningAnalyzer
from omnitab.tab_ocr.recognizer.smart_ocr import SmartTabOCR

def run_cycle(image_path: str):
    print("=" * 60)
    print("CYCLE: Deep Analysis & Improvement")
    print("=" * 60)
    
    # Load and process
    image = cv2.imread(image_path)
    ocr = SmartTabOCR()
    result = ocr.process(image)
    digits = result['digits']
    
    # Analyze
    analyzer = LearningAnalyzer()
    analysis = analyzer.analyze_digits(digits)
    
    print("\n[1] Basic Stats")
    print(f"    Total digits: {analysis.total_digits}")
    print(f"    Unique frets: {analysis.unique_frets}")
    
    print("\n[2] Fret Distribution (top 10)")
    for fret, count in sorted(analysis.fret_distribution.items(), key=lambda x: -x[1])[:10]:
        bar = "*" * min(count, 20)
        print(f"    Fret {fret:2d}: {count:3d} {bar}")
    
    print("\n[3] String Distribution")
    for string in range(1, 7):
        count = analysis.string_distribution.get(string, 0)
        bar = "*" * min(count, 20)
        print(f"    String {string}: {count:3d} {bar}")
    
    print("\n[4] Confidence Stats")
    for k, v in analysis.confidence_stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")
    
    print("\n[5] Suspicious Patterns")
    if analysis.suspicious_patterns:
        for p in analysis.suspicious_patterns:
            print(f"    [{p['severity'].upper()}] {p['description']}")
    else:
        print("    None found!")
    
    print("\n[6] Suggestions")
    for s in analysis.suggestions:
        print(f"    - {s}")
    
    print("\n[7] Attempt Comparison")
    comparison = analyzer.compare_attempts()
    for a in comparison.get('attempts', []):
        marker = " <-- BEST" if a == comparison.get('best') else ""
        print(f"    {a['id']} ({a['method']}): {a['gp5_notes']} notes{marker}")
    
    if 'improvement_potential' in comparison:
        ip = comparison['improvement_potential']
        print(f"\n[8] Improvement Potential")
        for k, v in ip.items():
            print(f"    {k}: {v}")
    
    return analysis, comparison


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_samples/images/page_1.png"
    run_cycle(path)
