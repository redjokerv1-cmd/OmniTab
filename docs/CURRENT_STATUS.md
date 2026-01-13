# OmniTab 현재 상태 보고서

**작성일**: 2026-01-13  
**버전**: 0.3.0 (실패)  
**목표**: TAB 이미지 → Guitar Pro 5 (.gp5) 변환

---

## 📊 프로젝트 개요

### 원래 목표
```
PDF/이미지 TAB 악보를 Guitar Pro 5 파일로 자동 변환
- 숫자 인식 (OCR)
- 튜닝/카포 자동 감지
- 마디/박자 구조 파싱
- GP5 파일 생성
```

### 실제 달성한 것
```
✅ 숫자 OCR: 148개 인식 (약 80% 정확도)
✅ 튜닝/카포 감지: 부분 작동 (카포 100%, 튜닝 50%)
✅ 코드 그룹핑: X 위치 기반 (0 problems)
⚠️ TAB 라인 감지: 1/3 시스템만 감지
❌ 줄 번호 매핑: 거의 작동 안 함
❌ 마디 구분: 작동 안 함
❌ 박자/리듬: 구현 안 됨
❌ GP5 생성: 쓸모없는 결과물
```

---

## 🔴 핵심 실패 원인

### 1. TAB 구조 파싱 실패

**문제**: TAB 악보의 6줄을 정확히 감지하지 못함

```
TAB 악보 구조:
  String 1: |--0--2--3--|--0--2--3--|
  String 2: |--1--3--0--|--1--3--0--|
  String 3: |--0--2--0--|--0--2--0--|
  String 4: |--2--0--0--|--2--0--0--|
  String 5: |--3--x--2--|--3--x--2--|
  String 6: |--x--x--3--|--x--x--3--|

필요한 정보:
  - 각 줄의 정확한 Y 좌표
  - 각 숫자가 어느 줄에 있는지
  - 마디 경계 (세로선 | 위치)

현재 상태:
  - 6줄 감지: 실패 (1/3 시스템만)
  - 줄 매핑: Y 위치 추정 (정확도 ~20%)
  - 마디 감지: 노이즈 포함 (84개 vs 실제 ~10개)
```

### 2. 잘못된 GP5 생성 로직

**문제**: 모든 음표를 1개 마디에 넣음

```python
# 현재 코드 (잘못됨)
measure = track.measures[0]  # 첫 번째 마디만 사용
for chord in all_chords:
    voice.beats.append(beat)  # 전부 한 마디에 추가

# 필요한 코드
for measure_chords in chords_per_measure:
    measure = create_new_measure()
    for chord in measure_chords[:4]:  # 4/4 = 4박
        measure.add_beat(chord)
```

### 3. 줄 번호 추정 실패

**문제**: Y 위치로 줄을 추정하는 로직이 완전히 틀림

```python
# 현재 코드 (완전 틀림)
def _estimate_string_from_y(self, note, chord):
    # 코드 내 상대 위치로 추정 → 완전 틀림
    idx = sorted_notes.index(note)
    return [1, 3, 5][idx]  # 임의 배정

# 필요한 코드
def get_string_for_note(self, note, tab_lines):
    # 실제 TAB 라인 Y 좌표와 비교
    for line in tab_lines:
        if abs(note.y - line.y_position) < line.tolerance:
            return line.string_num
```

---

## 📁 현재 코드 구조

```
omnitab/
├── __init__.py
├── gp5/
│   ├── __init__.py
│   └── writer.py              # GP5 파일 생성 (MIDI pitch 기반, 작동함)
├── notation/
│   ├── __init__.py
│   ├── detector.py            # 표기법 감지 (미완성)
│   └── normalizer.py          # 표기법 정규화 (미완성)
├── omr/
│   ├── __init__.py
│   └── engine.py              # OMR 엔진 (oemer 래퍼, 피아노용이라 TAB에 부적합)
└── tab_ocr/                   # TAB OCR 시스템 (주요 작업)
    ├── __init__.py
    ├── ocr_to_gp5.py          # OCR→GP5 변환 (❌ 실패)
    ├── pipeline.py            # 기본 파이프라인
    ├── models/
    │   ├── __init__.py
    │   ├── tab_note.py        # TabNote 데이터 모델
    │   ├── tab_chord.py       # TabChord 데이터 모델
    │   ├── tab_measure.py     # TabMeasure 데이터 모델
    │   └── tab_song.py        # TabSong 데이터 모델
    ├── parser/
    │   ├── __init__.py
    │   ├── chord_grouper.py   # 코드 그룹핑 (⚠️ 부분 작동)
    │   ├── measure_detector.py # 마디 감지 (❌ 미완성)
    │   └── timing_analyzer.py # 타이밍 분석 (❌ 미완성)
    ├── preprocessor/
    │   ├── __init__.py
    │   ├── image_loader.py    # 이미지 로드
    │   ├── line_detector.py   # 라인 감지 (❌ 부정확)
    │   └── region_detector.py # 영역 감지
    └── recognizer/
        ├── __init__.py
        ├── digit_ocr.py       # 기본 숫자 OCR
        ├── enhanced_ocr.py    # 개선된 OCR (✅ 148 digits, 80%)
        ├── header_detector.py # 헤더 감지 (⚠️ 카포 OK, 튜닝 부분)
        ├── improved_digit_separator.py  # 숫자 분리
        ├── line_remover.py    # 가로줄 제거 (✅ 작동)
        ├── position_mapper.py # 위치 매핑 (❌ 실패)
        ├── simple_binary_ocr.py # 단순 이진 OCR
        ├── symbol_ocr.py      # 기호 OCR (미완성)
        └── tab_line_detector.py # TAB 라인 감지 (❌ 1/3만)
```

---

## 🔍 시도한 방법들

### 1. OMR (Optical Music Recognition) 접근

**시도**: oemer 라이브러리로 악보 → MusicXML

**결과**: ❌ 실패
- oemer는 피아노 악보용
- TAB 악보를 피아노 악보로 오인식
- 완전히 잘못된 결과

### 2. 단순 OCR 접근

**시도**: EasyOCR로 전체 이미지에서 숫자 인식

**결과**: ⚠️ 부분 성공
- 숫자는 인식됨 (160개)
- 하지만 어느 줄인지 모름
- 마디 구분 불가

### 3. 가로줄 제거 + OCR

**시도**: 
1. 모폴로지 연산으로 TAB 가로줄 제거
2. 숫자만 남겨서 OCR

**결과**: ⚠️ 부분 성공
- 코드 그룹핑 개선 (0 problems)
- 하지만 줄 정보 손실
- 148 digits 인식

### 4. TAB 라인 감지 (수평 투영 - NEW!)

**시도**: 
1. 수평 투영 프로파일로 라인 찾기
2. 6개 등간격 라인 그룹 찾기

**결과**: ✅ 부분 성공 (2026-01-13 업데이트)
- 2/3 시스템 감지 (이전 1/3)
- **94.6% confidence**
- **정확한 Y 좌표** (14.8px 등간격)
- 3번째 시스템은 피크 불규칙으로 감지 실패

**코드**: `horizontal_projection.py`

### 5. 마디 경계 감지

**시도**: 수직선 감지로 마디 구분

**결과**: ❌ 실패
- 84개 "마디" 감지 (실제 ~10개)
- 노이즈가 마디로 오인식
- 필터링 필요

---

## 🎯 근본적인 문제

### 문제 1: TAB 구조의 복잡성

```
TAB 악보의 특성:
1. 6개의 수평선 (가로줄)
2. 선 위에 숫자 (프렛 번호)
3. 세로선 (마디 구분)
4. 특수 기호 (H, P, /, \, x, etc.)
5. 리듬 표기 (음표 꼬리, 빔)

문제:
- 가로줄이 숫자를 통과 → OCR 방해
- 숫자와 줄의 관계 파악 어려움
- 마디 경계와 다른 세로선 구분 어려움
```

### 문제 2: OCR 후 구조 복원의 어려움

```
OCR 결과: [('5', x=100, y=450), ('0', x=105, y=480), ...]

필요한 정보:
- '5'는 몇 번째 줄인가? → 줄 Y 좌표 필요
- '5'와 '0'은 같은 코드인가? → X 위치 비교
- 이것은 몇 번째 마디인가? → 마디 경계 필요

현재 상태:
- 줄 Y 좌표: 부정확 (1/3만 감지)
- 코드 그룹핑: 작동 (X 위치 기반)
- 마디 경계: 실패 (노이즈)
```

### 문제 3: 리듬/박자 정보 없음

```
TAB에서 리듬을 알 수 있는 방법:
1. 음표 꼬리/빔 (있으면)
2. 숫자 간격 (상대적)
3. 마디 내 위치

현재 상태:
- 음표 꼬리 감지: 미구현
- 숫자 간격 분석: 미구현
- 모든 음표를 4분음표로 처리 (틀림)
```

---

## 💡 가능한 해결책

### 방법 A: 전통적 이미지 처리 개선

```
필요한 작업:
1. TAB 라인 감지 알고리즘 개선
   - Hough 변환으로 직선 감지
   - 6줄 그룹 클러스터링 개선
   
2. 마디 경계 필터링
   - 길이 기반 필터 (짧은 선 제거)
   - 위치 기반 필터 (TAB 영역 내만)
   
3. 리듬 감지
   - 음표 꼬리/빔 감지
   - 없으면 간격으로 추정

예상 소요: 2-4주
성공 확률: 50-60%
```

### 방법 B: 딥러닝 기반 접근

```
필요한 작업:
1. TAB 악보 데이터셋 구축
2. 객체 감지 모델 훈련 (YOLO 등)
   - 숫자, 기호, 마디선 감지
3. 시퀀스 모델로 구조 파싱

예상 소요: 1-3개월
성공 확률: 70-80%
데이터 필요: 1000+ TAB 이미지
```

### 방법 C: 반자동 시스템

```
사용자가 직접 지정:
1. TAB 영역 선택
2. 6줄 위치 클릭
3. 마디 경계 클릭

시스템이 자동:
1. 숫자 OCR
2. 줄 매핑 (사용자 지정 기준)
3. GP5 생성

예상 소요: 1주
성공 확률: 90%+
단점: 수동 작업 필요
```

### 방법 D: 기존 도구 활용

```
TuxGuitar 또는 다른 도구가 지원하는 형식으로:
1. PDF → 이미지
2. 이미지 → MIDI (외부 도구)
3. MIDI → GP5 (TuxGuitar)

또는:
1. PDF에서 텍스트 기반 TAB 추출
2. 텍스트 TAB → GP5 변환

예상 소요: 1-2주
성공 확률: 가변적
의존성: 외부 도구
```

---

## 📈 작동하는 부분 (재사용 가능)

### 1. 숫자 OCR 파이프라인
```python
from omnitab.tab_ocr.recognizer.enhanced_ocr import EnhancedTabOCR

ocr = EnhancedTabOCR()
result = ocr.process_file("tab.png")
# 148 digits, 80% accuracy
```

### 2. 가로줄 제거
```python
from omnitab.tab_ocr.recognizer.line_remover import StaffLineRemover

remover = StaffLineRemover(kernel_length=40, repair_kernel=2)
clean_image = remover.process(image)
```

### 3. 헤더 감지 (카포/튜닝)
```python
from omnitab.tab_ocr.recognizer.header_detector import HeaderDetector

detector = HeaderDetector()
info = detector.detect(image)
# Capo: 2 (정확), Tuning: ['E', 'C', 'G', '?', '?', '?']
```

### 4. GP5 Writer (MIDI 기반)
```python
from omnitab.gp5.writer import GP5Writer

writer = GP5Writer(title="Song", tempo=120)
writer.write(notes_data, output_path)
# MIDI pitch + effects → GP5 (작동함)
```

---

## 🔧 테스트 데이터

### Yellow Jacket - Shaun Martin

```
파일: test_samples/images/page_1.png
튜닝: ①=E ②=C ③=G ④=D ⑤=G ⑥=C
카포: 2
템포: 65 BPM
박자: 4/4

OCR 결과:
- 148 digits recognized
- 86 chords grouped
- 5 systems detected (실제 3개)

GP5 결과:
- ❌ 모든 음표가 1마디에
- ❌ 줄 번호 틀림
- ❌ 사용 불가
```

---

## 📋 결론

### 현재 상태 요약

```
OmniTab v0.3.0

성공:
- 숫자 OCR (80%)
- 카포 감지 (100%)
- 가로줄 제거

실패:
- TAB 라인 감지
- 줄 번호 매핑
- 마디 구분
- GP5 생성

결론: MVP 실패
원인: TAB 구조 파싱의 복잡성 과소평가
```

### 다음 단계 권장

```
단기 (1-2주):
→ 방법 C (반자동 시스템) 시도
  - 사용자가 줄/마디 지정
  - 시스템이 OCR + GP5 생성
  - 최소한 작동하는 결과물

중기 (1-2개월):
→ 방법 A (이미지 처리 개선)
  - TAB 라인 감지 알고리즘 개선
  - 마디 경계 필터링
  - 완전 자동화

장기 (3개월+):
→ 방법 B (딥러닝)
  - 데이터셋 구축
  - 모델 훈련
  - 높은 정확도
```

---

## 📚 참고 자료

### 관련 프로젝트
- [Audiveris](https://github.com/Audiveris/audiveris) - OMR (표준 악보)
- [oemer](https://github.com/BreezeWhite/oemer) - OMR (피아노)
- [TuxGuitar](http://tuxguitar.com.ar/) - GP 편집기

### 기술 스택
- Python 3.11+
- OpenCV
- EasyOCR
- PyGuitarPro

### 테스트 환경
- Windows 10/11
- GPU: 선택적 (EasyOCR)

---

*마지막 업데이트: 2026-01-13*
