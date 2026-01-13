# Gemini Vision API 사용법

## 1. API 키 발급

1. https://makersuite.google.com/app/apikey 접속
2. "Create API Key" 클릭
3. 키 복사

## 2. 환경 변수 설정

```powershell
# Windows PowerShell (일회성)
$env:GOOGLE_API_KEY = "your-api-key-here"

# Windows (영구)
setx GOOGLE_API_KEY "your-api-key-here"
```

## 3. 코드 사용법

```python
from omnitab.tab_ocr.gemini_analyzer import GeminiTabAnalyzer

# 초기화
analyzer = GeminiTabAnalyzer(api_key="YOUR_API_KEY")
# 또는 환경변수 사용
analyzer = GeminiTabAnalyzer()

# TAB 이미지 분석
result = analyzer.analyze("test_samples/images/page_1.png")

print(result)
```

## 4. 결과 형식

```json
{
  "measures": [
    {
      "number": 1,
      "beats": [
        {
          "duration": "quarter",
          "notes": [
            {"string": 1, "fret": 10, "technique": null},
            {"string": 2, "fret": 0, "technique": null},
            {"string": 3, "fret": 12, "technique": "hammer-on"}
          ]
        }
      ]
    }
  ],
  "tuning": ["E", "B", "G", "D", "A", "E"],
  "capo": 2,
  "tempo": 120
}
```

## 5. GP5로 변환

```python
from omnitab.tab_ocr.gemini_analyzer import gemini_to_gp5_notes

gp5_notes = gemini_to_gp5_notes(result)
# Duration 값:
#   "whole" -> -2
#   "half" -> -1  
#   "quarter" -> 0
#   "eighth" -> 1
#   "sixteenth" -> 2
```

## 6. 현재 OCR + Gemini 조합

```python
# OCR로 노트 위치 감지 (정확)
# Gemini로 리듬 분석 (AI)

from omnitab.tab_ocr.smart_to_gp5 import SmartToGp5
from omnitab.tab_ocr.gemini_analyzer import GeminiTabAnalyzer

# 1. OCR로 노트 감지
converter = SmartToGp5()
ocr_result = converter.convert("image.png", "output.gp5")

# 2. Gemini로 리듬 추가
analyzer = GeminiTabAnalyzer()
rhythm_result = analyzer.analyze("image.png")

# 3. 조합하여 최종 GP5 생성
# (추후 구현 예정)
```

## 비용

- Gemini 1.5 Flash: 무료 티어 있음 (분당 60 요청)
- 이미지당 약 0.001$ 미만
