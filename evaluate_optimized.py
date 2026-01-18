"""Evaluate OptimizedAnalyzer accuracy"""

import json
from omnitab.tab_ocr.optimized_analyzer import OptimizedAnalyzer


def load_ground_truth(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_beat(gt_beat: dict, pred_beat: dict) -> dict:
    gt_notes = {(n["s"], n["f"]) for n in gt_beat.get("notes", [])}
    pred_notes = set()
    
    for n in pred_beat.get("notes", []):
        s = n.get("string")
        f = n.get("fret")
        if s and f is not None:
            pred_notes.add((s, f))
    
    correct = gt_notes & pred_notes
    missing = gt_notes - pred_notes
    extra = pred_notes - gt_notes
    
    return {
        "correct": len(correct),
        "missing": len(missing),
        "extra": len(extra),
        "gt_count": len(gt_notes),
    }


def evaluate():
    print("Loading ground truth...")
    gt = load_ground_truth("test_samples/ground_truth/page_1.json")
    
    print("Analyzing with OptimizedAnalyzer...")
    analyzer = OptimizedAnalyzer()
    pred = analyzer.analyze(
        "test_samples/images/page_1.png",
        calibration="S1:10, S2:0, S3:12, S4:10",
        debug=True
    )
    
    # Flatten beats
    gt_beats = []
    for m in gt["measures"]:
        gt_beats.extend(m.get("beats", []))
    
    pred_beats = []
    for m in pred.get("measures", []):
        pred_beats.extend(m.get("beats", []))
    
    print()
    print("=" * 60)
    print("EVALUATION - OptimizedAnalyzer")
    print("=" * 60)
    print(f"GT Beats: {len(gt_beats)}, Pred Beats: {len(pred_beats)}")
    
    total_correct = 0
    total_missing = 0
    total_extra = 0
    total_gt_notes = 0
    perfect_beats = 0
    
    min_beats = min(len(gt_beats), len(pred_beats))
    
    for i in range(min_beats):
        result = compare_beat(gt_beats[i], pred_beats[i])
        total_correct += result["correct"]
        total_missing += result["missing"]
        total_extra += result["extra"]
        total_gt_notes += result["gt_count"]
        
        if result["missing"] == 0 and result["extra"] == 0:
            perfect_beats += 1
    
    # Remaining GT beats
    for i in range(min_beats, len(gt_beats)):
        gt_notes = len(gt_beats[i].get("notes", []))
        total_missing += gt_notes
        total_gt_notes += gt_notes
    
    accuracy = (total_correct / total_gt_notes * 100) if total_gt_notes > 0 else 0
    precision = (total_correct / (total_correct + total_extra) * 100) if (total_correct + total_extra) > 0 else 0
    
    print()
    print("[Results]")
    print(f"  Total GT Notes: {total_gt_notes}")
    print(f"  Correct: {total_correct}")
    print(f"  Missing: {total_missing}")
    print(f"  Extra: {total_extra}")
    print()
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  Perfect Beats: {perfect_beats}/{min_beats}")
    
    print()
    print("=" * 60)
    print(f"FINAL ACCURACY: {accuracy:.1f}%")
    print("=" * 60)
    
    return accuracy


if __name__ == "__main__":
    evaluate()
