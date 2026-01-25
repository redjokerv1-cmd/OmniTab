# OmniTab - Music Knowledge Graph 계획

**작성일**: 2026-01-25
**기반**: Semantica 분석 (universal-devkit/research/semantica-integration-analysis.md)

---

## 목표

OCR/딥러닝 인식 결과를 **음악 이론 지식 그래프**로 검증하여:
- 물리적으로 불가능한 프렛 보정
- 화성학 기반 코드 추론
- OCR 정확도 향상

---

## 핵심 아이디어

```
OCR 인식: 30프렛
    ↓ 물리적 검증
기타 최대 24프렛 → 오류!
    ↓ 유사 숫자 추론
30 → 3 (가장 유사한 유효값)
    ↓ 화성학 검증
앞뒤 코드 C-G-? → Am 확률 높음
    ↓ 최종 보정
3프렛 (Am 코드 구성음)
```

---

## MusicKnowledgeGraph 구조

```python
# omnitab/knowledge/music_kg.py

class MusicKnowledgeGraph:
    """음악 이론 지식 그래프"""
    
    # 물리적 제약
    GUITAR_FRETS = 24
    GUITAR_STRINGS = 6
    BASS_FRETS = 24
    BASS_STRINGS = 4
    
    # 음계 매핑 (6번줄 기준)
    STANDARD_TUNING = {
        6: "E2", 5: "A2", 4: "D3", 
        3: "G3", 2: "B3", 1: "E4"
    }
    
    # 화성학 규칙
    CHORD_PROGRESSIONS = {
        "C": ["G", "Am", "F", "Dm", "Em"],
        "G": ["D", "Em", "C", "Am", "Bm"],
        "D": ["A", "Bm", "G", "Em"],
        "A": ["E", "F#m", "D", "Bm"],
        "E": ["B", "C#m", "A", "F#m"],
        # ...
    }
    
    # 코드-프렛 매핑
    CHORD_SHAPES = {
        "C": [(5,3), (4,2), (2,1)],  # (줄, 프렛)
        "G": [(6,3), (5,2), (1,3)],
        "Am": [(5,0), (4,2), (3,2), (2,1)],
        # ...
    }
```

---

## 검증 메서드

### 1. 물리적 검증

```python
def validate_fret(self, string: int, fret: int) -> Tuple[bool, int]:
    """프렛 유효성 검사 및 보정
    
    Returns:
        (is_valid, corrected_fret)
    """
    if fret > self.GUITAR_FRETS:
        # OCR 오류 추정: 30 → 3, 25 → 2 or 5
        if fret >= 30:
            corrected = fret % 10  # 30→0, 31→1, ...
        elif fret > 24:
            corrected = fret - 20  # 25→5, 26→6, ...
        return False, corrected
    return True, fret

def validate_position(self, positions: List[Tuple[int, int]]) -> bool:
    """손가락 물리적 가능성 검사
    
    - 4프렛 이상 스트레칭 경고
    - 같은 프렛 다른 줄은 바레 코드로 가능
    """
    frets = [p[1] for p in positions if p[1] > 0]
    if not frets:
        return True
    span = max(frets) - min(frets)
    return span <= 4  # 일반적인 손가락 범위
```

### 2. 화성학 검증

```python
def suggest_chord(self, prev_chords: List[str], detected_notes: List[str]) -> str:
    """화성학 기반 코드 추론
    
    Args:
        prev_chords: 이전 마디 코드들
        detected_notes: OCR로 인식된 음들
        
    Returns:
        가장 가능성 높은 코드
    """
    if not prev_chords:
        return self._guess_from_notes(detected_notes)
    
    last_chord = prev_chords[-1]
    candidates = self.CHORD_PROGRESSIONS.get(last_chord, [])
    
    # 후보 중 detected_notes와 가장 일치하는 코드
    scores = {}
    for chord in candidates:
        chord_notes = self._get_chord_notes(chord)
        overlap = len(set(detected_notes) & set(chord_notes))
        scores[chord] = overlap
    
    return max(scores, key=scores.get)
```

### 3. 컨텍스트 기반 보정

```python
def correct_with_context(
    self, 
    measure: List[Dict],  # 현재 마디 데이터
    prev_measure: List[Dict],
    next_measure: List[Dict]
) -> List[Dict]:
    """전후 마디 컨텍스트로 보정"""
    
    corrected = []
    for note in measure:
        # 1. 물리적 검증
        is_valid, new_fret = self.validate_fret(note['string'], note['fret'])
        
        if not is_valid:
            # 2. 화성학 힌트 적용
            context_chord = self._detect_chord(prev_measure)
            suggested = self.suggest_chord([context_chord], [])
            
            # 3. 가장 적합한 프렛 선택
            new_fret = self._best_fret_for_chord(note['string'], suggested)
        
        corrected.append({**note, 'fret': new_fret})
    
    return corrected
```

---

## 통합 파이프라인

```python
# omnitab/core/tab_processor.py

class TabProcessor:
    def __init__(self):
        self.ocr = TabOCR()
        self.music_kg = MusicKnowledgeGraph()
    
    def process(self, image) -> TabData:
        # 1. OCR 인식
        raw_tab = self.ocr.recognize(image)
        
        # 2. 지식 그래프 검증 & 보정
        validated_tab = self.music_kg.validate_and_correct(raw_tab)
        
        # 3. 신뢰도 점수 계산
        confidence = self._calculate_confidence(raw_tab, validated_tab)
        
        return TabData(tab=validated_tab, confidence=confidence)
```

---

## 구현 우선순위

1. [ ] `MusicKnowledgeGraph` 기본 클래스
2. [ ] `validate_fret()` - 프렛 범위 검증
3. [ ] `CHORD_PROGRESSIONS` 데이터 구축
4. [ ] `suggest_chord()` - 화성학 추론
5. [ ] `correct_with_context()` - 컨텍스트 보정
6. [ ] 기존 OCR 파이프라인 통합

---

## 예상 성과

| 지표 | 현재 | 목표 |
|------|------|------|
| 프렛 인식 오류 | ~5% | <1% |
| 코드 추론 정확도 | N/A | 80%+ |
| 전체 TAB 정확도 | ~90% | 95%+ |

---

*참고: universal-devkit/research/semantica-integration-analysis.md*
