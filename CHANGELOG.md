# Changelog

All notable changes to OmniTab will be documented in this file.

---

## [0.4.0] - 2026-01-13

### 🎉 BREAKTHROUGH: Smart OCR - 64% Improvement!

#### 핵심 성과
```
┌────────────────┬──────────┬───────────┐
│ Metric         │ 이전     │ Smart OCR │
├────────────────┼──────────┼───────────┤
│ Digits         │ 91       │ 120 (+32%)│
│ 2-digit frets  │ 0        │ 42        │
│ GP5 Notes      │ 61       │ 100 (+64%)│
└────────────────┴──────────┴───────────┘
```

#### 창의적 해결책
- **문제**: Contour 기반 OCR이 2자리 숫자를 분리 ("10" → "1" + "0")
- **해결**: 슬라이딩 윈도우 방식으로 TAB 라인 스캔
- **결과**: 10-24 프렛 100% confidence로 인식

#### Added
- `smart_ocr.py` - 슬라이딩 윈도우 TAB OCR
- `smart_to_gp5.py` - Smart OCR 기반 GP5 변환
- `tab_region_ocr.py` - TAB 영역 전용 OCR
- `learning/` - 학습 DB (모든 시도 추적)
  - SQLite 기반 저장
  - 시도별 정확도 비교
  - 오류 패턴 수집

#### Learning DB
```
2개 시도 기록됨:
- 기존 방식: 91 digits, 61 GP5 notes
- Smart OCR: 120 digits, 100 GP5 notes ← 최고 성과
```

#### 교훈
1. 기존 방법이 안되면 창의적으로 접근
2. 데이터를 쌓으면서 개선 (DB화)
3. 문제의 근본 원인 파악이 핵심

---

## [0.3.1] - 2026-01-13

### 🔴 Status: MVP Failed - Core Features Incomplete

#### 문서화
- `docs/CURRENT_STATUS.md` - 현재 상태 솔직한 분석
- `docs/ARCHITECTURE.md` - 실제 아키텍처 (작동/실패 표시)
- `README.md` - 현실적으로 업데이트

#### 핵심 실패 원인
```
❌ TAB 6줄 감지: 1/3만 성공
❌ 줄 번호 매핑: 정확도 ~20%
❌ 마디 구분: 노이즈 문제
❌ GP5 생성: 사용 불가
```

#### 작동하는 부분
```
✅ 숫자 OCR: 148개, 80%
✅ 카포 감지: 100%
✅ 가로줄 제거: 작동
✅ GP5Writer (MIDI 기반): 작동
```

#### 교훈
1. TAB 구조 파싱은 단순 OCR로 불가능
2. 6줄 감지가 핵심 (없으면 모든 게 틀림)
3. 반자동 접근이 현실적

---

## [0.3.0] - 2026-01-13

### 🚀 Major: Complete TAB Image to GP5 Pipeline (❌ 실패)

#### Added
- **OcrToGp5Converter** (`ocr_to_gp5.py`)
  - Full pipeline: TAB image → GP5 file
  - Custom tuning support (6 note names)
  - Capo position support
  - Auto-detect or manual override

#### Usage
```python
from omnitab.tab_ocr.ocr_to_gp5 import convert_tab_to_gp5

result = convert_tab_to_gp5(
    image_path="tab.png",
    output_path="output.gp5",
    title="Song Name",
    tempo=120,
    tuning=['E', 'C', 'G', 'D', 'G', 'C'],  # Optional
    capo=2  # Optional
)
```

#### Yellow Jacket Test
```
Input:  page_1.png
Output: yellow_jacket_correct.gp5
Beats:  86
Tuning: E C G D G C (alternate)
Capo:   2
```

---

## [0.2.2] - 2026-01-13

### ✨ Feature: Auto-detect Tuning & Capo

#### Added
- **HeaderDetector** (`header_detector.py`)
  - Scans image header for tuning/capo info
  - Detects: `①=E ②=C` format, `Tuning: DADGAD` format
  - Detects: `Capo. fret 2` or `Capo 3`
  - Flags non-standard tuning for manual verification

#### Results (Yellow Jacket test)
```
Tuning: ['E', 'C', 'G', '?', '?', '?']  ← Partial detection
Capo: 2 ✅
Standard: False ✅
[!] Alternate tuning - manual verification recommended
```

#### Integration
- EnhancedTabOCR now includes header detection
- Result dict contains `header` with tuning/capo info

---

## [0.2.1] - 2026-01-13

### 🔧 Fix: Chord Grouping Problem

#### Problem
- Horizontal staff lines connected digits → OCR grouped them as one blob
- Result: Chords with 9-16 notes (impossible on 6-string guitar)

#### Solution: Hybrid Approach
1. **Staff Line Removal** (`line_remover.py`)
   - Morphological operations to detect/remove horizontal lines
   - `kernel=40`, `repair=2` (optimal parameters)

2. **Tight Chord Grouping**
   - Reduced `x_threshold` from 25 to 5
   - Prevents adjacent digits from merging

3. **EnhancedTabOCR** (`enhanced_ocr.py`)
   - Combines line removal + optimized grouping
   - Single class with best parameters

#### Results
| Metric | Before | After |
|--------|--------|-------|
| Problems (>6 notes) | 5 | **0** ✅ |
| Max notes/chord | 16 | **5** |
| Chords detected | 47 | **86** |
| Confidence | 80% | 79.6% |

---

## [0.2.0] - 2026-01-13

### 🚀 Major: TAB OCR Phase 1 Complete

#### Added
- **TAB OCR System** - Complete image-to-TAB recognition pipeline
  - `omnitab/tab_ocr/` - New OCR module
  - Data models: `TabNote`, `TabChord`, `TabMeasure`, `TabSong`
  - Preprocessor: `ImageLoader`, `RegionDetector`, `LineDetector`
  - Recognizer: `DigitOCR`, `SymbolOCR`, `PositionMapper`
  - Parser: `ChordGrouper`, `MeasureDetector`, `TimingAnalyzer`
  - Pipeline: `TabOCRPipeline`

- **SimpleBinaryOCR** - Optimized for black/white sheet music
  - Key insight: Sheet music is B&W → simple binary threshold works best
  - Small digit scaling (3x) for better recognition
  - Result: 160 digits recognized (vs 1 with basic OCR)

- **improved_digit_separator.py** - Phase 1 implementation
  - CLAHE enhancement
  - Contour-based digit separation
  - Individual digit OCR with EasyOCR

#### Changed
- **OCR Parameters optimized for small digits**
  - `min_digit_width`: 5 → 3
  - `min_digit_height`: 8 → 5
  - `scale_factor`: 3 (3x magnification)
  - `min_confidence`: 0.3 → 0.2

#### Performance
| Metric | Before | After |
|--------|--------|-------|
| Digits recognized | 1 | 160 |
| TAB systems | 0 | 6 |
| Chords | 0 | 47 |
| Avg confidence | - | 80% |

#### Key Insight
> "악보는 거의 대부분 흑백이다" → 단순 이진화가 복잡한 전처리보다 효과적

---

## [0.1.0] - 2026-01-12

### Added
- **GP5Writer** - Guitar Pro 5 file generation
  - PyGuitarPro integration
  - Note/chord/rest support
  - Effects: hammer-on, pull-off, slide, bend, vibrato, harmonic

- **NotationDetector** - Detect notation types from notes
  - 50+ notation patterns supported
  - Confidence scoring

- **NotationNormalizer** - Normalize detected notations

- **Basic OMR integration** - oemer for optical music recognition

### Tests
- 22 unit tests passing

---

## [0.0.1] - 2026-01-11

### Added
- Initial project setup
- Repository created: `git@github.com:redjokerv1-cmd/OmniTab.git`
- Basic project structure
