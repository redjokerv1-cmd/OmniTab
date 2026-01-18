"""Evaluate LabeledTabAnalyzer accuracy"""

import json
from omnitab.tab_ocr.labeled_tab_analyzer import LabeledTabAnalyzer


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
        "missing_notes": list(missing),
        "extra_notes": list(extra)
    }


def evaluate():
    print("Loading ground truth...")
    gt = load_ground_truth("test_samples/ground_truth/page_1.json")
    
    print("Analyzing with LabeledTabAnalyzer...")
    analyzer = LabeledTabAnalyzer()
    pred = analyzer.analyze("test_samples/images/page_1.png")
    
    gt_measures = gt["measures"]
    pred_measures = pred.get("measures", [])
    
    print()
    print("=" * 60)
    print("EVALUATION - LabeledTabAnalyzer")
    print("=" * 60)
    print(f"GT Measures: {len(gt_measures)}, Pred Measures: {len(pred_measures)}")
    
    # Flatten beats
    gt_beats = []
    for m in gt_measures:
        gt_beats.extend(m.get("beats", []))
    
    pred_beats = []
    for m in pred_measures:
        pred_beats.extend(m.get("beats", []))
    
    print(f"GT Beats: {len(gt_beats)}, Pred Beats: {len(pred_beats)}")
    
    total_correct = 0
    total_missing = 0
    total_extra = 0
    total_gt_notes = 0
    
    min_beats = min(len(gt_beats), len(pred_beats))
    beat_results = []
    
    for i in range(min_beats):
        result = compare_beat(gt_beats[i], pred_beats[i])
        total_correct += result["correct"]
        total_missing += result["missing"]
        total_extra += result["extra"]
        total_gt_notes += result["gt_count"]
        beat_results.append(result)
    
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
    
    print()
    print("[First 8 Beat Comparisons]")
    for i, result in enumerate(beat_results[:8]):
        status = "OK" if result["missing"] == 0 and result["extra"] == 0 else "ERR"
        print(f"  Beat {i+1}: {status} correct={result['correct']}/{result['gt_count']}", end="")
        if result["missing_notes"]:
            print(f" miss={result['missing_notes'][:2]}", end="")
        if result["extra_notes"]:
            print(f" extra={result['extra_notes'][:2]}", end="")
        print()
    
    print()
    print("=" * 60)
    print(f"FINAL ACCURACY: {accuracy:.1f}%")
    print("=" * 60)
    
    return accuracy


if __name__ == "__main__":
    evaluate()
