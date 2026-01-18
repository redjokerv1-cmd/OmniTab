# OmniTab 현재 상태 보고서

**작성일**: 2026-01-18  
**버전**: 0.7.1  
**목표**: TAB 이미지 → Guitar Pro 5 (.gp5) 변환

---

## 📊 프로젝트 개요

### 현재 달성 상태

```
✅ Gemini Vision 기반 TAB 분석 (권장 방식)
✅ 줄 번호 인식 (1-4번 줄 정확)
✅ 튜닝/카포 감지 (100% 정확)
✅ 마디 구분 (5-6 마디 정확)
✅ GP5 파일 생성 (291개 노트)
✅ 테크닉 인식 (H, P, AH, muted 등)
✅ 웹 UI + REST API
⚠️ 5-6번 줄 빈 줄 처리 (부분 작동)
⚠️ 리듬 정확도 (Gemini 종속)
```

---

## 🟢 최근 해결된 문제

### 1. GP5 노트 저장 버그 (v0.7.0)

**문제**: Gemini가 129개 노트를 감지해도 GP5 파일에는 0개 저장

**원인**: 
```python
# 잘못된 코드
track = gp.Track(song)           # 새 트랙 생성
song.tracks.append(track)        # 2번째 트랙이 됨

# PyGuitarPro는 Song() 생성 시 자동으로 Track을 1개 만듦
# append하면 2번째가 되어 저장 후 무시됨
```

**해결**:
```python
# 올바른 코드
track = song.tracks[0]           # 기존 트랙 사용
track.name = "Acoustic Guitar"   # 직접 수정
track.measures.clear()           # 기존 measures 삭제
```

### 2. Gemini 줄 번호 오인식 (v0.7.1)

**문제**: TAB 1-4번 줄이 완전히 잘못된 줄 번호로 인식됨

**원인**: 프롬프트에 TAB 읽기 규칙이 명확하지 않음

**해결**: 프롬프트에 상세 예시 추가
```
TAB:
e|--5--  ← String 1 (TOP line), fret 5
B|--7--  ← String 2
G|--0--  ← String 3
D|-----  ← String 4 (skip)
A|--2--  ← String 5
E|--3--  ← String 6 (BOTTOM line)
```

### 3. technique 리스트 처리 (v0.7.1)

**문제**: Gemini가 technique을 리스트로 반환하면 AttributeError

**해결**: `_apply_technique()` 메서드에 리스트 처리 로직 추가

---

## 📁 현재 코드 구조

```
omnitab/
├── __init__.py
├── api/
│   ├── main.py              # FastAPI REST API ✅
│   └── static/
│       └── index.html       # 웹 프론트엔드 ✅
├── gp5/
│   └── writer.py            # GP5 파일 생성 (MIDI pitch 기반) ✅
├── learning/
│   └── db.py                # SQLite 학습 DB ✅
├── notation/
│   ├── detector.py          # 표기법 감지
│   └── normalizer.py        # 표기법 정규화
└── tab_ocr/                  # TAB OCR 시스템
    ├── gemini_analyzer.py       # Gemini Vision API ✅ (핵심!)
    ├── gemini_only_converter.py # Gemini 전용 변환기 ✅ (권장!)
    ├── batch_converter.py       # 일괄 변환 ✅
    ├── complete_converter.py    # 하이브리드 변환
    ├── sliced_gemini_converter.py # 줄별 분할 변환
    ├── models/
    │   ├── tab_note.py
    │   ├── tab_chord.py
    │   ├── tab_measure.py
    │   └── tab_song.py
    ├── parser/
    │   ├── chord_grouper.py
    │   ├── measure_detector.py
    │   └── timing_analyzer.py
    ├── preprocessor/
    │   ├── image_loader.py
    │   ├── line_detector.py
    │   ├── region_detector.py
    │   └── score_slicer.py      # 줄/마디 분할 ✅
    └── recognizer/
        ├── enhanced_ocr.py      # EasyOCR 기반 ⚠️
        ├── header_detector.py   # 헤더 감지 ✅
        ├── line_remover.py      # 가로줄 제거 ✅
        └── smart_ocr.py         # 슬라이딩 윈도우 OCR
```

---

## 🎯 권장 사용 방법

### 방법 1: Gemini Only (권장)

```python
from omnitab.tab_ocr.gemini_only_converter import GeminiOnlyConverter

converter = GeminiOnlyConverter()
result = converter.convert(
    image_path='tab.png',
    output_path='output.gp5',
    title='Song Name',
    tempo=120
)
```

**장점**:
- 가장 정확한 줄 번호 인식
- 리듬/박자 자동 분석
- 테크닉 자동 감지

### 방법 2: 웹 UI

```bash
uvicorn omnitab.api.main:app --reload
# http://localhost:8000
```

---

## 📈 테스트 결과

### Yellow Jacket - page_1.png

| 지표 | v0.6.0 | v0.7.0 | v0.7.1 |
|------|--------|--------|--------|
| 노트 수 | 294 | 129 | **291** ✅ |
| 마디 수 | 6 | 6 | **5-6** ✅ |
| 줄 번호 | ❌ | ❌ | **✅ 1-4번 정확** |
| 튜닝 | ✅ | ✅ | ✅ |
| 카포 | ✅ 2 | ✅ 2 | ✅ 2 |
| GP5 저장 | ❌ 0개 | ✅ 129개 | ✅ 291개 |

---

## 🔧 남은 과제

### 단기 (1-2주)

1. **빈 줄 처리 개선**
   - 현재: 빈 줄에 다른 비트의 숫자가 잘못 포함됨
   - 목표: 비어있는 줄은 건너뛰기

2. **리듬 정확도 향상**
   - Gemini의 리듬 분석 결과 검증 로직 추가
   - 음표 스템/빔 분석으로 보정

### 중기 (1개월)

1. **다중 페이지 안정화**
   - 페이지 간 마디 번호 연속성
   - 반복 기호 처리

2. **학습 DB 활용**
   - 사용자 수정 기록으로 정확도 향상
   - 오류 패턴 학습

---

## 📚 참고 자료

- [PyGuitarPro 문서](https://pyguitarpro.readthedocs.io/)
- [PYGUITARPRO_REFERENCE.md](./PYGUITARPRO_REFERENCE.md)
- [google-genai SDK](https://ai.google.dev/)

---

*마지막 업데이트: 2026-01-18*
