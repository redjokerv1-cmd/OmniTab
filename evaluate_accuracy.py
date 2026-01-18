"""
Evaluate OmniTab accuracy against ground truth
"""

import json
from pathlib import Path
from omnitab.tab_ocr.gemini_analyzer import GeminiTabAnalyzer


def load_ground_truth(path: str) -> dict:
    """Load ground truth JSON"""
    with open(path) as f:
        return json.load(f)


def analyze_image(image_path: str) -> dict:
    """Analyze image with Gemini"""
    analyzer = GeminiTabAnalyzer()
    return analyzer.analyze(image_path)


def compare_beat(gt_beat: dict, pred_beat: dict) -> dict:
    """Compare a single beat"""
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
        "pred_count": len(pred_notes),
        "missing_notes": list(missing),
        "extra_notes": list(extra)
    }


def evaluate(gt_path: str, image_path: str) -> dict:
    """Full evaluation"""
    print("Loading ground truth...")
    gt = load_ground_truth(gt_path)
    
    print("Analyzing image with Gemini...")
    pred = analyze_image(image_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Compare metadata
    print("\n[Metadata]")
    print(f"  Tuning - GT: {gt['metadata']['tuning']}, Pred: {pred.get('tuning')}")
    print(f"  Capo   - GT: {gt['metadata']['capo']}, Pred: {pred.get('capo')}")
    print(f"  Tempo  - GT: {gt['metadata']['tempo']}, Pred: {pred.get('tempo')}")
    
    tuning_match = gt['metadata']['tuning'] == pred.get('tuning')
    capo_match = gt['metadata']['capo'] == pred.get('capo')
    
    # Compare measures
    gt_measures = gt['measures']
    pred_measures = pred.get('measures', [])
    
    print(f"\n[Measures]")
    print(f"  GT: {len(gt_measures)}, Pred: {len(pred_measures)}")
    
    total_correct = 0
    total_missing = 0
    total_extra = 0
    total_gt_notes = 0
    
    errors_by_measure = []
    
    min_measures = min(len(gt_measures), len(pred_measures))
    
    for i in range(min_measures):
        gt_m = gt_measures[i]
        pred_m = pred_measures[i]
        
        gt_beats = gt_m.get("beats", [])
        pred_beats = pred_m.get("beats", [])
        
        measure_correct = 0
        measure_missing = 0
        measure_extra = 0
        measure_errors = []
        
        min_beats = min(len(gt_beats), len(pred_beats))
        
        for j in range(min_beats):
            result = compare_beat(gt_beats[j], pred_beats[j])
            measure_correct += result["correct"]
            measure_missing += result["missing"]
            measure_extra += result["extra"]
            total_gt_notes += result["gt_count"]
            
            if result["missing"] > 0 or result["extra"] > 0:
                measure_errors.append({
                    "beat": j + 1,
                    "missing": result["missing_notes"],
                    "extra": result["extra_notes"]
                })
        
        # Count remaining GT beats as missing
        for j in range(min_beats, len(gt_beats)):
            gt_notes = len(gt_beats[j].get("notes", []))
            measure_missing += gt_notes
            total_gt_notes += gt_notes
        
        total_correct += measure_correct
        total_missing += measure_missing
        total_extra += measure_extra
        
        errors_by_measure.append({
            "measure": i + 1,
            "gt_beats": len(gt_beats),
            "pred_beats": len(pred_beats),
            "correct": measure_correct,
            "missing": measure_missing,
            "extra": measure_extra,
            "errors": measure_errors[:3]  # Top 3 errors
        })
    
    # Calculate accuracy
    if total_gt_notes > 0:
        accuracy = (total_correct / total_gt_notes) * 100
    else:
        accuracy = 0
    
    precision = total_correct / (total_correct + total_extra) * 100 if (total_correct + total_extra) > 0 else 0
    recall = total_correct / total_gt_notes * 100 if total_gt_notes > 0 else 0
    
    print(f"\n[Note-level Results]")
    print(f"  Total GT Notes: {total_gt_notes}")
    print(f"  Correct:        {total_correct}")
    print(f"  Missing:        {total_missing}")
    print(f"  Extra:          {total_extra}")
    print(f"\n  Accuracy:  {accuracy:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  Recall:    {recall:.1f}%")
    
    print(f"\n[Per-Measure Breakdown]")
    for m in errors_by_measure:
        status = "OK" if m["missing"] == 0 and m["extra"] == 0 else "ERR"
        print(f"  M{m['measure']}: {status} beats={m['pred_beats']}/{m['gt_beats']}, correct={m['correct']}, missing={m['missing']}, extra={m['extra']}")
        if m["errors"]:
            for e in m["errors"][:2]:
                if e["missing"]:
                    print(f"      Beat {e['beat']} missing: {e['missing']}")
                if e["extra"]:
                    print(f"      Beat {e['beat']} extra: {e['extra']}")
    
    print("\n" + "=" * 60)
    print(f"FINAL ACCURACY: {accuracy:.1f}%")
    print("=" * 60)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "total_gt_notes": total_gt_notes,
        "correct": total_correct,
        "missing": total_missing,
        "extra": total_extra,
        "tuning_match": tuning_match,
        "capo_match": capo_match,
        "errors_by_measure": errors_by_measure
    }


if __name__ == "__main__":
    result = evaluate(
        gt_path="test_samples/ground_truth/page_1.json",
        image_path="test_samples/images/page_1.png"
    )
