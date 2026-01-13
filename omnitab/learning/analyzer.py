"""
Learning Analyzer - Analyze OCR results and find improvement opportunities

This module:
1. Compares OCR output with expected values
2. Identifies error patterns
3. Suggests improvements
4. Tracks progress across iterations
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import json

from .db import LearningDB
from .models import ErrorPattern


@dataclass
class AnalysisResult:
    """Result of analyzing OCR output"""
    total_digits: int
    unique_frets: List[int]
    fret_distribution: Dict[int, int]
    string_distribution: Dict[int, int]
    suspicious_patterns: List[Dict]
    confidence_stats: Dict[str, float]
    suggestions: List[str]


class LearningAnalyzer:
    """
    Analyze OCR results to find patterns and improvements.
    
    Deep learning cycle:
    1. Run OCR → 2. Analyze results → 3. Find patterns → 4. Improve → repeat
    """
    
    def __init__(self, db: LearningDB = None):
        self.db = db or LearningDB()
    
    def analyze_digits(self, digits: List) -> AnalysisResult:
        """Analyze a list of detected digits"""
        if not digits:
            return AnalysisResult(
                total_digits=0,
                unique_frets=[],
                fret_distribution={},
                string_distribution={},
                suspicious_patterns=[],
                confidence_stats={},
                suggestions=["No digits detected - check image quality"]
            )
        
        # Basic stats
        frets = [int(d.value) if hasattr(d, 'value') else int(d['value']) for d in digits]
        strings = [int(d.string) if hasattr(d, 'string') else int(d.get('string', 0)) for d in digits]
        confs = [float(d.confidence) if hasattr(d, 'confidence') else float(d.get('confidence', 0)) for d in digits]
        
        fret_dist = dict(Counter(frets))
        string_dist = dict(Counter(strings))
        
        # Find suspicious patterns
        suspicious = []
        
        # Pattern 1: Frets > 19 (rare in fingerstyle)
        high_frets = [f for f in frets if f > 19]
        if high_frets:
            suspicious.append({
                'type': 'high_fret',
                'description': f'Frets > 19 detected: {set(high_frets)}',
                'count': len(high_frets),
                'severity': 'medium'
            })
        
        # Pattern 2: Uneven string distribution
        if string_dist:
            max_string = max(string_dist.values())
            min_string = min(string_dist.values())
            if max_string > min_string * 3:
                suspicious.append({
                    'type': 'uneven_strings',
                    'description': f'Uneven string distribution: {string_dist}',
                    'severity': 'low'
                })
        
        # Pattern 3: Low confidence
        low_conf_count = sum(1 for c in confs if c < 0.5)
        if low_conf_count > len(confs) * 0.2:
            suspicious.append({
                'type': 'low_confidence',
                'description': f'{low_conf_count}/{len(confs)} digits have low confidence',
                'severity': 'high'
            })
        
        # Confidence stats
        conf_stats = {
            'min': min(confs) if confs else 0,
            'max': max(confs) if confs else 0,
            'avg': sum(confs) / len(confs) if confs else 0,
            'below_50': sum(1 for c in confs if c < 0.5),
            'above_90': sum(1 for c in confs if c >= 0.9)
        }
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            fret_dist, string_dist, suspicious, conf_stats
        )
        
        return AnalysisResult(
            total_digits=len(digits),
            unique_frets=sorted(set(frets)),
            fret_distribution=fret_dist,
            string_distribution=string_dist,
            suspicious_patterns=suspicious,
            confidence_stats=conf_stats,
            suggestions=suggestions
        )
    
    def _generate_suggestions(self,
                              fret_dist: Dict,
                              string_dist: Dict,
                              suspicious: List,
                              conf_stats: Dict) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []
        
        # Check for missing 2-digit numbers
        has_10_plus = any(f >= 10 for f in fret_dist.keys())
        if not has_10_plus:
            suggestions.append(
                "No 2-digit frets (10+) detected - consider using Smart OCR"
            )
        
        # Check confidence
        if conf_stats['avg'] < 0.7:
            suggestions.append(
                f"Average confidence is low ({conf_stats['avg']:.2f}) - try increasing scale_factor"
            )
        
        # Check string coverage
        if string_dist:
            missing_strings = [s for s in range(1, 7) if s not in string_dist]
            if missing_strings:
                suggestions.append(
                    f"Strings {missing_strings} have no notes - check line detection"
                )
        
        # Suspicious patterns
        for s in suspicious:
            if s['severity'] == 'high':
                suggestions.append(f"HIGH: {s['description']}")
        
        if not suggestions:
            suggestions.append("OCR quality looks good!")
        
        return suggestions
    
    def compare_attempts(self, attempt_ids: List[str] = None) -> Dict:
        """Compare multiple OCR attempts to find best method"""
        if attempt_ids:
            attempts = [self.db.get_attempt(id) for id in attempt_ids]
        else:
            attempts = self.db.get_all_attempts(limit=10)
        
        if not attempts:
            return {'error': 'No attempts found'}
        
        comparison = []
        for a in attempts:
            comparison.append({
                'id': a.id[:8],
                'method': a.settings.get('method', 'unknown'),
                'mapped': a.mapped_digits,
                'total': a.total_digits,
                'mapping_rate': a.mapped_digits / max(a.total_digits, 1),
                'gp5_notes': a.gp5_notes,
                'avg_confidence': a.avg_confidence,
                'suspicious': a.suspicious_count
            })
        
        # Find best
        best = max(comparison, key=lambda x: x['gp5_notes'])
        
        return {
            'attempts': comparison,
            'best': best,
            'improvement_potential': self._calculate_improvement_potential(comparison)
        }
    
    def _calculate_improvement_potential(self, comparison: List[Dict]) -> Dict:
        """Calculate potential for improvement"""
        if len(comparison) < 2:
            return {'status': 'Need more attempts for comparison'}
        
        gp5_notes = [c['gp5_notes'] for c in comparison]
        best = max(gp5_notes)
        worst = min(gp5_notes)
        
        return {
            'best_gp5_notes': best,
            'worst_gp5_notes': worst,
            'improvement_range': f"{((best - worst) / max(worst, 1)) * 100:.1f}%",
            'recommendation': 'Use Smart OCR method' if best > worst else 'Methods are similar'
        }
    
    def find_error_patterns(self, digits: List, expected: List = None) -> List[ErrorPattern]:
        """Find recurring error patterns"""
        patterns = []
        
        # Pattern 1: Repeated same fret in sequence (likely OCR duplication)
        if len(digits) >= 2:
            for i in range(len(digits) - 1):
                d1 = digits[i]
                d2 = digits[i + 1]
                
                v1 = d1.value if hasattr(d1, 'value') else d1['value']
                v2 = d2.value if hasattr(d2, 'value') else d2['value']
                x1 = d1.x if hasattr(d1, 'x') else d1['x']
                x2 = d2.x if hasattr(d2, 'x') else d2['x']
                
                if v1 == v2 and abs(x1 - x2) < 15:
                    patterns.append(ErrorPattern(
                        id=f"dup_{i}",
                        pattern_type='duplicate',
                        description=f'Duplicate fret {v1} at X={x1:.0f} and {x2:.0f}',
                        x=x1,
                        y=d1.y if hasattr(d1, 'y') else d1['y']
                    ))
        
        return patterns
    
    def save_insight(self, insight_type: str, description: str, data: Dict):
        """Save a learned insight to the database"""
        self.db.save_insight(insight_type, description, data)
    
    def get_improvement_history(self) -> List[Dict]:
        """Get history of improvements across attempts"""
        attempts = self.db.get_all_attempts(limit=50)
        
        history = []
        for a in sorted(attempts, key=lambda x: x.timestamp):
            history.append({
                'timestamp': a.timestamp.isoformat(),
                'method': a.settings.get('method', 'unknown'),
                'gp5_notes': a.gp5_notes,
                'mapping_rate': a.mapped_digits / max(a.total_digits, 1)
            })
        
        return history


# CLI
if __name__ == '__main__':
    analyzer = LearningAnalyzer()
    
    print("=== Attempt Comparison ===")
    comparison = analyzer.compare_attempts()
    
    for a in comparison.get('attempts', []):
        print(f"\n{a['id']} ({a['method']}):")
        print(f"  Mapped: {a['mapped']}/{a['total']} ({a['mapping_rate']*100:.1f}%)")
        print(f"  GP5 Notes: {a['gp5_notes']}")
        print(f"  Confidence: {a['avg_confidence']:.2f}")
    
    if 'best' in comparison:
        print(f"\n=== Best Method ===")
        print(f"Method: {comparison['best']['method']}")
        print(f"GP5 Notes: {comparison['best']['gp5_notes']}")
