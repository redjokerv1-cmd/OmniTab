# Score Slicing Guide (악보 이미지 분할)

## 개요

악보 한 페이지를 통째로 AI에게 보내면:
- 마디 순서 혼동 가능
- 작은 숫자 놓칠 수 있음
- 토큰 사용량 증가

**해결책:** 악보를 시스템(줄) 단위로 잘라서 순차 처리

## 아키텍처

```
PDF → 페이지 이미지 → 줄별 분할 → Gemini (각 줄) → JSON 합치기 → GP5
```

## 핵심 알고리즘: Morphological Dilation

```python
import cv2
import numpy as np

def slice_score(image_path):
    # 1. 이미지 로드
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 이진화 (반전)
    thresh = cv2.threshold(gray, 0, 255, 
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # 3. Dilation - 핵심!
    # 가로 100, 세로 30 커널로 음표들을 하나의 덩어리로 뭉침
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 30))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    # 4. 윤곽선 검출
    contours, _ = cv2.findContours(dilation, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Y좌표로 정렬 (위에서 아래로)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    # 6. 잘라서 저장
    results = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        
        # 노이즈 필터링 (너무 작은 영역 제외)
        if w < img.shape[1] * 0.5 or h < 50:
            continue
        
        # Padding 추가
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        
        cropped = img[y1:y2, x1:x2]
        path = f"system_{i:03d}.jpg"
        cv2.imwrite(path, cropped)
        results.append(path)
    
    return results
```

## 커널 사이즈 튜닝

| 문제 | 해결 |
|------|------|
| 표준악보와 TAB이 따로 잘림 | 세로(30) → 50으로 증가 |
| 옆 마디가 짤림 | 가로(100) → 150으로 증가 |
| 페이지 번호가 같이 잘림 | 필터링 조건 강화 |

## 전체 파이프라인 예시

```python
from score_slicer import ScoreSlicer
from gemini_analyzer import GeminiTabAnalyzer
from gp5_converter import GP5Converter

def process_score(pdf_path):
    # 1. PDF → 이미지
    images = convert_pdf_to_images(pdf_path)
    
    # 2. 각 페이지를 줄별로 분할
    slicer = ScoreSlicer()
    all_systems = []
    for img in images:
        systems = slicer.slice_image(img)
        all_systems.extend(systems)
    
    # 3. 각 줄을 Gemini로 분석
    analyzer = GeminiTabAnalyzer()
    all_measures = []
    for system in all_systems:
        result = analyzer.analyze(system)
        all_measures.extend(result.get('measures', []))
    
    # 4. GP5 생성
    converter = GP5Converter()
    converter.create_song(all_measures)
    converter.save("output.gp5")
```

## OmniTab에서의 현재 구현

OmniTab은 현재 **HorizontalProjection**을 사용하여 TAB 시스템을 감지합니다:

```python
# omnitab/tab_ocr/recognizer/horizontal_projection.py
class HorizontalProjection:
    def detect_systems(self, image):
        # 수평 프로젝션으로 6줄 TAB 시스템 감지
        ...
```

이는 Dilation 방식과 다른 접근이지만 동일한 목표를 달성합니다.

## 언제 분할이 필요한가?

| 상황 | 권장 |
|------|------|
| 1-2 페이지 악보 | 전체 페이지 처리 OK |
| 복잡한 핑거스타일 | 줄별 분할 권장 |
| 작은 숫자가 많음 | 줄별 분할 권장 |
| API 비용 절약 | 전체 페이지 처리 |
| 최고 정확도 필요 | 줄별 분할 |

## 참고
- OpenCV Morphological Operations: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
