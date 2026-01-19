"""
Gemini Vision을 사용한 TAB 자동 라벨링

실제 TAB 이미지를 Gemini에게 보내서 각 숫자의 위치와 프렛 번호를 추출
YOLO annotation 형식으로 저장
"""

import os
import re
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from PIL import Image
import io

# .env 파일 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class GeminiAutoLabeler:
    """Gemini Vision API를 사용한 TAB 자동 라벨링"""
    
    CLASSES = [str(i) for i in range(25)] + ['h', 'p', 'x', 'harmonic']
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required. Set it in environment or pass to constructor.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def label_image(self, image_path: str, output_dir: str = None) -> Dict:
        """
        TAB 이미지를 분석하여 annotation 생성
        
        Args:
            image_path: TAB 이미지 경로
            output_dir: 출력 디렉토리 (기본: 이미지와 같은 폴더)
        
        Returns:
            annotation 결과 딕셔너리
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else image_path.parent
        
        # 이미지 로드
        image = Image.open(image_path)
        width, height = image.size
        
        print(f"[Gemini] Analyzing {image_path.name} ({width}x{height})...")
        
        # Gemini에게 요청
        prompt = self._create_prompt(width, height)
        
        response = self.model.generate_content([prompt, image])
        
        # 응답 파싱
        try:
            result = self._parse_response(response.text, width, height)
            print(f"[Gemini] Found {len(result['annotations'])} annotations")
        except Exception as e:
            print(f"[Error] Failed to parse response: {e}")
            print(f"[Raw Response] {response.text[:500]}...")
            return {"error": str(e), "raw_response": response.text}
        
        # 저장
        self._save_annotations(result, image_path, output_dir)
        
        return result
    
    def _create_prompt(self, width: int, height: int) -> str:
        """Gemini 프롬프트 생성"""
        return f"""You are analyzing a guitar TAB (tablature) image.

**Image dimensions: {width} x {height} pixels**

**Task:** Find ALL fret numbers (0-24) and technique symbols in this TAB image.

**Rules:**
1. TAB has 6 horizontal lines (strings). String 1 (thinnest, high E) is at TOP, String 6 (thickest, low E) is at BOTTOM.
2. Numbers on the lines indicate fret positions (0 = open string, 1-24 = fret numbers)
3. Look for technique symbols: h (hammer-on), p (pull-off), x (mute), harmonics

**Output Format (JSON only, no markdown):**
{{
  "annotations": [
    {{
      "fret": 0,
      "x": 162,
      "y": 594,
      "string": 1
    }},
    {{
      "fret": 5,
      "x": 245,
      "y": 610,
      "string": 2
    }}
  ]
}}

**Important:**
- x, y are pixel coordinates of the CENTER of each number
- fret is the number value (0-24) or "h", "p", "x" for techniques
- string is 1-6 (1=top line, 6=bottom line)
- Include EVERY number you can see, even if partially visible
- Be precise with coordinates

Return ONLY the JSON, no explanation."""
    
    def _parse_response(self, response_text: str, width: int, height: int) -> Dict:
        """Gemini 응답 파싱 (잘린 JSON 복구 지원)"""
        text = response_text.strip()
        
        # 마크다운 코드 블록 제거
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
            else:
                # 끝 마커가 없으면 시작부터 끝까지
                text = text[start:].strip()
        elif text.startswith("```"):
            lines = text.split("\n")
            # 끝 마커 찾기
            end_idx = -1
            for i, line in enumerate(lines):
                if i > 0 and line.strip() == "```":
                    end_idx = i
                    break
            if end_idx > 0:
                text = "\n".join(lines[1:end_idx])
            else:
                text = "\n".join(lines[1:])
        
        # 잘린 JSON 복구 - 각 annotation 객체를 개별 파싱
        annotations = []
        
        # 개별 annotation 객체 추출
        pattern = r'\{\s*"fret"\s*:\s*("[^"]*"|\d+)\s*,\s*"x"\s*:\s*(\d+)\s*,\s*"y"\s*:\s*(\d+)\s*,\s*"string"\s*:\s*(\d+)\s*\}'
        matches = re.findall(pattern, text)
        
        for match in matches:
            fret_raw, x, y, string = match
            
            # fret 값 처리
            if fret_raw.startswith('"'):
                fret = fret_raw.strip('"')
            else:
                fret = int(fret_raw)
            
            annotations.append({
                "fret": fret,
                "x": int(x),
                "y": int(y),
                "string": int(string)
            })
        
        if annotations:
            data = {"annotations": annotations}
        else:
            # 정규식 매칭 실패 시 기존 방식 시도
            try:
                # JSON 복구 - 마지막 완전한 객체까지
                if not text.endswith("}"):
                    last_complete = text.rfind("},")
                    if last_complete > 0:
                        text = text[:last_complete + 1] + "\n  ]\n}"
                
                data = json.loads(text)
            except json.JSONDecodeError:
                # 부분 파싱 시도
                partial_pattern = r'"fret"\s*:\s*("[^"]*"|\d+)[^}]*"x"\s*:\s*(\d+)[^}]*"y"\s*:\s*(\d+)'
                partial_matches = re.findall(partial_pattern, text)
                
                for match in partial_matches:
                    fret_raw, x, y = match
                    if fret_raw.startswith('"'):
                        fret = fret_raw.strip('"')
                    else:
                        fret = int(fret_raw)
                    annotations.append({
                        "fret": fret,
                        "x": int(x),
                        "y": int(y),
                        "string": 1  # 기본값
                    })
                
                if annotations:
                    data = {"annotations": annotations}
                else:
                    raise ValueError(f"Could not parse any annotations from response")
        
        annotations = []
        for item in data.get("annotations", []):
            fret = item.get("fret")
            x = item.get("x", 0)
            y = item.get("y", 0)
            string_num = item.get("string", 0)
            
            # 클래스 결정
            if isinstance(fret, int):
                class_name = str(fret)
                class_id = fret if fret < 25 else 24
            elif isinstance(fret, str):
                class_name = fret.lower()
                class_id = self.CLASSES.index(class_name) if class_name in self.CLASSES else 0
            else:
                continue
            
            # YOLO 형식으로 변환 (normalized)
            x_center = x / width
            y_center = y / height
            box_width = 20 / width  # 기본 박스 크기
            box_height = 20 / height
            
            annotations.append({
                "class_id": class_id,
                "class_name": class_name,
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
                "pixel_x": x,
                "pixel_y": y,
                "string": string_num
            })
        
        return {
            "width": width,
            "height": height,
            "annotations": annotations
        }
    
    def _save_annotations(self, result: Dict, image_path: Path, output_dir: Path):
        """annotation 저장"""
        base_name = image_path.stem
        
        # YOLO txt 파일
        txt_path = output_dir / f"{base_name}.txt"
        with open(txt_path, 'w') as f:
            for ann in result['annotations']:
                line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                f.write(line)
        print(f"[Saved] {txt_path}")
        
        # JSON 백업
        json_path = output_dir / f"{base_name}_annotation.json"
        with open(json_path, 'w') as f:
            json.dump({
                "image": str(image_path),
                "width": result['width'],
                "height": result['height'],
                "annotations": result['annotations'],
                "source": "gemini_auto_labeler"
            }, f, indent=2)
        print(f"[Saved] {json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini Auto Labeler for TAB images")
    parser.add_argument("--image", "-i", required=True, help="Path to TAB image")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env)")
    
    args = parser.parse_args()
    
    labeler = GeminiAutoLabeler(api_key=args.api_key)
    result = labeler.label_image(args.image, args.output)
    
    if "error" not in result:
        print(f"\n[OK] Success! {len(result['annotations'])} annotations created.")
    else:
        print(f"\n[ERROR] {result['error']}")


if __name__ == "__main__":
    main()
