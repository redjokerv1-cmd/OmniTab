# 🎸 OmniTab

> **상태: 🔴 MVP 실패 - 핵심 기능 미완성**

TAB 악보 이미지를 Guitar Pro 5 (.gp5) 파일로 변환하는 도구 (개발 중)

---

## ⚠️ 현재 상태

```
✅ 작동하는 것:
- 숫자 OCR: 148개 인식 (80% 정확도)
- 카포 감지: 100%
- 가로줄 제거

❌ 작동하지 않는 것:
- TAB 6줄 감지 (1/3만 성공)
- 줄 번호 매핑 (정확도 ~20%)
- 마디 구분 (노이즈 문제)
- GP5 생성 (사용 불가한 결과물)

결론: 이미지 → GP5 변환 불가
```

**상세 보고서**: [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)

---

## 📊 문제 분석

### 핵심 실패 원인

```
TAB 악보 구조:
  String 1: |--0--2--3--|--0--2--3--|
  String 2: |--1--3--0--|--1--3--0--|
  ...

필요한 정보:
1. 각 줄의 Y 좌표 (6줄)
2. 각 숫자가 어느 줄에 있는지
3. 마디 경계 (세로선)
4. 박자/리듬

현재 감지 가능:
- 숫자 값 ✅
- 대략적인 X, Y 위치 ⚠️
- 그 외 전부 ❌
```

---

## 🛠️ 설치

```bash
# 가상환경
python -m venv venv
venv\Scripts\activate  # Windows

# 의존성
pip install -r requirements.txt
```

---

## 💻 사용 가능한 기능

### 1. 숫자 OCR (작동)

```python
from omnitab.tab_ocr.recognizer.enhanced_ocr import EnhancedTabOCR

ocr = EnhancedTabOCR()
result = ocr.process_file("tab.png")

print(f"숫자: {len(result['digits'])}개")
print(f"코드: {len(result['systems'])}개")
```

### 2. 헤더 감지 (부분 작동)

```python
from omnitab.tab_ocr.recognizer.header_detector import HeaderDetector

detector = HeaderDetector()
info = detector.detect_file("tab.png")

print(f"Capo: {info.capo}")  # 정확
print(f"Tuning: {info.tuning}")  # 부분
```

### 3. GP5 생성 (MIDI 기반 - 작동)

```python
from omnitab.gp5.writer import GP5Writer

# MIDI pitch 기반 (직접 지정 필요)
notes_data = [
    {"type": "note", "pitch": 64, "duration": 4},
    {"type": "chord", "pitches": [64, 59, 55], "duration": 4},
]

writer = GP5Writer(title="Song", tempo=120)
writer.write(notes_data, "output.gp5")
```

---

## 📁 프로젝트 구조

```
OmniTab/
├── docs/
│   ├── CURRENT_STATUS.md    # 현재 상태 (실패 분석)
│   └── ARCHITECTURE.md      # 아키텍처
├── omnitab/
│   ├── gp5/                  # GP5 생성 (✅ 작동)
│   ├── notation/             # 표기법 (미완성)
│   ├── omr/                  # OMR (미사용)
│   └── tab_ocr/              # TAB OCR (⚠️ 부분)
│       ├── recognizer/       # OCR 모듈
│       ├── parser/           # 파싱 (❌ 실패)
│       └── preprocessor/     # 전처리
├── test_samples/
│   ├── images/               # 테스트 이미지
│   └── output/               # 출력 파일
├── tests/                    # 22개 테스트
├── CHANGELOG.md
└── requirements.txt
```

---

## 🔮 해결 방향

### 방법 A: 이미지 처리 개선 (2-4주)
- Hough 변환으로 TAB 라인 감지
- 마디 경계 필터링 개선
- 성공 확률: 50-60%

### 방법 B: 딥러닝 (1-3개월)
- TAB 데이터셋 구축
- 객체 감지 모델 훈련
- 성공 확률: 70-80%

### 방법 C: 반자동 (1주)
- 사용자가 줄/마디 지정
- 시스템이 OCR + GP5 생성
- 성공 확률: 90%+

**권장**: 방법 C로 먼저 작동하는 결과물 만들기

---

## 📋 테스트

```bash
# 단위 테스트 (22개 - 데이터 모델)
pytest tests/ -v

# OCR 테스트
python -m omnitab.tab_ocr.recognizer.enhanced_ocr image.png

# 헤더 감지 테스트
python -m omnitab.tab_ocr.recognizer.header_detector image.png
```

---

## 📚 테스트 데이터

**Yellow Jacket - Shaun Martin**
- 튜닝: E C G D G C
- 카포: 2
- 결과: OCR 148개, GP5 생성 실패

---

## 📝 교훈

1. **TAB 구조 파싱은 어렵다** - 단순 OCR로는 부족
2. **6줄 감지가 핵심** - 이게 없으면 모든 게 틀림
3. **반자동이 현실적** - 완전 자동화는 어려움

---

## 📄 라이선스

MIT

---

*마지막 업데이트: 2026-01-13*
*상태: 개발 중단 (MVP 실패)*
