# OmniTab 아키텍처

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OmniTab Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [입력: PDF/이미지]                                                  │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               1. PREPROCESSOR                                 │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ✅ ImageLoader      PDF → 이미지 변환                        │   │
│  │  ✅ LineRemover      TAB 가로줄 제거 (OCR 개선용)             │   │
│  │  ❌ RegionDetector   TAB 영역 감지 (부정확)                   │   │
│  │  ❌ LineDetector     6줄 감지 (1/3만 성공)                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               2. RECOGNIZER                                   │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ✅ EnhancedOCR      숫자 인식 (148 digits, 80%)              │   │
│  │  ✅ HeaderDetector   카포/튜닝 감지 (카포 100%, 튜닝 50%)     │   │
│  │  ❌ PositionMapper   줄 번호 매핑 (정확도 ~20%)               │   │
│  │  ⚠️ SymbolOCR       특수 기호 (H, P 등) (미완성)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               3. PARSER                                       │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ⚠️ ChordGrouper     코드 그룹핑 (X 위치 기반, 작동)          │   │
│  │  ❌ MeasureDetector  마디 감지 (노이즈 문제)                  │   │
│  │  ❌ TimingAnalyzer   박자 분석 (미구현)                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               4. GP5 WRITER                                   │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  ✅ GP5Writer        GP5 파일 생성 (MIDI 기반 - 작동)         │   │
│  │  ❌ OcrToGp5         OCR→GP5 변환 (구조 파싱 실패)            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  [출력: .gp5 파일]                                                   │
│                                                                      │
│  현재 상태: ❌ 사용 불가 (구조 파싱 실패)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 컴포넌트 상세

### 1. Preprocessor (전처리)

| 컴포넌트 | 파일 | 상태 | 설명 |
|---------|------|------|------|
| ImageLoader | `image_loader.py` | ✅ | PDF→이미지, 이미지 로드 |
| LineRemover | `line_remover.py` | ✅ | 가로줄 제거 (kernel=40, repair=2) |
| RegionDetector | `region_detector.py` | ❌ | TAB 영역 감지 실패 |
| TabLineDetector | `tab_line_detector.py` | ❌ | 6줄 감지 (1/3 성공) |

### 2. Recognizer (인식)

| 컴포넌트 | 파일 | 상태 | 설명 |
|---------|------|------|------|
| EnhancedTabOCR | `enhanced_ocr.py` | ✅ | 숫자 OCR (148 digits, 80%) |
| SimpleBinaryOCR | `simple_binary_ocr.py` | ✅ | 단순 이진화 OCR |
| HeaderDetector | `header_detector.py` | ⚠️ | 카포 100%, 튜닝 50% |
| PositionMapper | `position_mapper.py` | ❌ | 줄 매핑 실패 |
| SymbolOCR | `symbol_ocr.py` | ⚠️ | 미완성 |

### 3. Parser (파싱)

| 컴포넌트 | 파일 | 상태 | 설명 |
|---------|------|------|------|
| ChordGrouper | `chord_grouper.py` | ⚠️ | X 위치 기반 그룹핑 (작동) |
| MeasureDetector | `measure_detector.py` | ❌ | 노이즈 문제로 실패 |
| TimingAnalyzer | `timing_analyzer.py` | ❌ | 미구현 |

### 4. GP5 Writer

| 컴포넌트 | 파일 | 상태 | 설명 |
|---------|------|------|------|
| GP5Writer | `gp5/writer.py` | ✅ | MIDI pitch → GP5 (작동) |
| OcrToGp5 | `ocr_to_gp5.py` | ❌ | 구조 파싱 실패로 무용 |

---

## 데이터 흐름

### 현재 (실패)

```
이미지 → 숫자 OCR → (X,Y) 좌표만 → 줄 추정 실패 → GP5 쓰레기
```

### 필요한 것

```
이미지 → TAB 구조 감지 → 6줄 Y 좌표 → 숫자 OCR
                           ↓
                       줄 매핑 → 마디 감지 → GP5 생성
```

---

## 핵심 누락 로직

### 1. TAB 6줄 감지

```python
# 현재 (실패)
def _detect_tab_systems(self, binary):
    # 수평 투영 → 라인 찾기 → 6개 그룹핑
    # 문제: 1/3만 감지

# 필요한 것
def detect_tab_lines(self, image):
    # 1. Hough 변환으로 모든 수평선 감지
    # 2. 선 길이/위치로 필터링
    # 3. Y 좌표로 클러스터링
    # 4. 6개씩 그룹핑 (등간격)
    # 5. 각 줄의 정확한 Y 좌표 반환
```

### 2. 숫자→줄 매핑

```python
# 현재 (틀림)
def estimate_string(note, chord):
    idx = sorted_notes.index(note)
    return [1,3,5][idx]  # 임의 배정

# 필요한 것
def get_string_number(note, tab_lines):
    for line in tab_lines:
        if abs(note.y - line.y) < tolerance:
            return line.string_num
    return None  # 매핑 실패
```

### 3. 마디 감지

```python
# 현재 (노이즈)
def detect_measures(binary):
    v_kernel = (1, 20)  # 너무 작음
    # 84개 "마디" 감지

# 필요한 것
def detect_measures(binary, tab_regions):
    # 1. TAB 영역 내에서만 검색
    # 2. 충분히 긴 세로선만 (6줄 통과)
    # 3. 일정 간격 필터링
    # 4. 시작/끝 마디선 구분
```

---

## 의존성

```
omnitab/
├── opencv-python    이미지 처리
├── numpy            수치 연산
├── easyocr          숫자 OCR
├── PyGuitarPro      GP5 생성
├── pdf2image        PDF 변환
└── pillow           이미지 처리
```

---

## 테스트

### 단위 테스트

```bash
pytest tests/test_tab_ocr.py -v
# 22 tests passed (데이터 모델만)
```

### 통합 테스트

```bash
python -m omnitab.tab_ocr.ocr_to_gp5 image.png output.gp5
# 결과: 사용 불가한 GP5 파일
```

---

## 파일 구조

```
G:\Study\AI\OmniTab\
├── docs/
│   ├── ARCHITECTURE.md         # 이 파일
│   ├── CURRENT_STATUS.md       # 현재 상태 보고서
│   └── TAB_OCR_ARCHITECTURE.md # 원본 설계 (이상적)
├── omnitab/
│   ├── gp5/                    # GP5 생성
│   ├── notation/               # 표기법 처리
│   ├── omr/                    # OMR (미사용)
│   └── tab_ocr/                # TAB OCR (주요)
├── test_samples/
│   ├── images/                 # 테스트 이미지
│   └── output/                 # 출력 파일
├── tests/                      # 테스트
├── CHANGELOG.md                # 변경 이력
├── README.md                   # 프로젝트 설명
└── requirements.txt            # 의존성
```

---

*마지막 업데이트: 2026-01-13*
