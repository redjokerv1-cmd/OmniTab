# Changelog

All notable changes to OmniTab will be documented in this file.

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
