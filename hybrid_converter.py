"""
Hybrid TAB to GP5 Converter

YOLO의 Domain Gap 문제를 해결하기 위해 Gemini Vision API 활용
+ 기존 구조 분석 (시스템 분리, 마디선 감지) 결합

아키텍처:
1. 이미지 전처리 (시스템 분리, 크롭)
2. Gemini Vision으로 각 시스템 분석 (프렛 + 리듬)
3. 구조 정보 결합 (마디선, 박자표)
4. GP5 생성
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import guitarpro as gp

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Warning] google-generativeai not installed")


class HybridConverter:
    """
    Gemini Vision + 구조 분석 하이브리드 변환기
    """
    
    # Gemini 프롬프트 (개선된 버전)
    ANALYSIS_PROMPT = """Analyze this guitar TAB image and extract ALL notes precisely.

CRITICAL RULES:
1. TAB has 6 horizontal lines. Count from TOP to BOTTOM:
   - Line 1 (TOP) = String 1 (thinnest, high E)
   - Line 2 = String 2 (B)
   - Line 3 = String 3 (G)
   - Line 4 = String 4 (D)
   - Line 5 = String 5 (A)
   - Line 6 (BOTTOM) = String 6 (thickest, low E)

2. Numbers ON each line are fret numbers (0-24)
3. Numbers vertically aligned = played simultaneously (chord)
4. Read LEFT to RIGHT for time order

5. Look at the STAFF NOTATION above TAB for rhythm:
   - Filled note head + stem = quarter note or shorter
   - Hollow note head = half note or longer
   - Flags/beams on stem = eighth/sixteenth notes
   - Dot after note = 1.5x duration

Return JSON format:
{
  "tempo": 120,
  "time_signature": "4/4",
  "measures": [
    {
      "number": 1,
      "beats": [
        {
          "duration": "quarter",  // whole, half, quarter, eighth, sixteenth
          "notes": [
            {"string": 1, "fret": 0},
            {"string": 2, "fret": 2}
          ]
        }
      ]
    }
  ]
}

IMPORTANT:
- Count EVERY number on the TAB lines
- Group vertically aligned numbers as one beat
- Use staff notation for rhythm, TAB for fret numbers
- Do NOT confuse staff note heads with TAB numbers"""

    def __init__(self, api_key: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai required")
        
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        print("[Hybrid] Gemini initialized")
    
    def _detect_tab_systems(self, image: np.ndarray) -> List[Dict]:
        """TAB 시스템 영역 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        row_sums = np.sum(horizontal, axis=1)
        threshold = image.shape[1] * 0.15 * 255
        
        line_rows = np.where(row_sums > threshold)[0]
        
        if len(line_rows) == 0:
            return []
        
        # 그룹화
        lines = []
        current_start = line_rows[0]
        
        for i in range(1, len(line_rows)):
            if line_rows[i] - line_rows[i-1] > 5:
                lines.append((current_start + line_rows[i-1]) // 2)
                current_start = line_rows[i]
        lines.append((current_start + line_rows[-1]) // 2)
        
        # 6줄 그룹 찾기
        systems = []
        i = 0
        
        while i < len(lines) - 5:
            group = lines[i:i+6]
            spacings = [group[j+1] - group[j] for j in range(5)]
            avg_spacing = sum(spacings) / 5
            
            if max(spacings) - min(spacings) < 15 and 10 < avg_spacing < 40:
                # 오선보 포함하여 크롭 (TAB 위로 더 확장)
                y_start = max(0, int(group[0] - avg_spacing * 6))  # 오선보 포함
                y_end = min(image.shape[0], int(group[-1] + avg_spacing * 2))
                
                systems.append({
                    'index': len(systems),
                    'y_start': y_start,
                    'y_end': y_end,
                    'tab_lines': [float(y) for y in group]
                })
                i += 6
            else:
                i += 1
        
        return systems
    
    def _analyze_with_gemini(self, image_path: str) -> Dict:
        """Gemini Vision으로 TAB 분석"""
        import PIL.Image
        
        img = PIL.Image.open(image_path)
        
        response = self.model.generate_content(
            [self.ANALYSIS_PROMPT, img],
            generation_config={"temperature": 0.1}
        )
        
        # JSON 파싱
        text = response.text
        
        # JSON 블록 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON parse error: {e}")
            print(f"Raw text: {text[:500]}")
            return {"measures": []}
    
    def convert(self, image_path: str, output_path: str,
                title: str = "Hybrid TAB", tempo: int = 120) -> str:
        """
        이미지 → GP5 변환
        """
        print("=" * 70)
        print("Hybrid Converter (Gemini Vision + Structure Analysis)")
        print("=" * 70)
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Error] Cannot load: {image_path}")
            return None
        
        h, w = image.shape[:2]
        print(f"\n[1] Image: {w}x{h}")
        
        # TAB 시스템 감지
        print("\n[2] Detecting TAB systems...")
        systems = self._detect_tab_systems(image)
        print(f"    Found {len(systems)} systems")
        
        if not systems:
            print("    No systems found, analyzing full image")
            systems = [{'index': 0, 'y_start': 0, 'y_end': h}]
        
        # 각 시스템을 Gemini로 분석
        all_measures = []
        
        for sys in systems:
            print(f"\n[3] Analyzing system {sys['index'] + 1}...")
            
            # 시스템 크롭
            crop = image[sys['y_start']:sys['y_end'], :]
            temp_path = f"temp_system_{sys['index']}.png"
            cv2.imwrite(temp_path, crop)
            
            # Gemini 분석
            try:
                result = self._analyze_with_gemini(temp_path)
                measures = result.get("measures", [])
                
                # 마디 번호 조정
                offset = len(all_measures)
                for m in measures:
                    m['number'] = offset + m.get('number', 1)
                
                all_measures.extend(measures)
                print(f"    Found {len(measures)} measures")
                
            except Exception as e:
                print(f"    [Error] Gemini analysis failed: {e}")
            
            # 임시 파일 삭제
            Path(temp_path).unlink(missing_ok=True)
        
        print(f"\n[4] Total measures: {len(all_measures)}")
        
        if not all_measures:
            print("[Error] No measures extracted!")
            return None
        
        # GP5 생성
        print("\n[5] Creating GP5...")
        song = self._create_gp5(all_measures, title, tempo)
        
        gp.write(song, output_path)
        
        # 통계
        total_beats = sum(
            len(m.get('beats', [])) for m in all_measures
        )
        total_notes = sum(
            len(b.get('notes', []))
            for m in all_measures
            for b in m.get('beats', [])
        )
        
        print(f"\n[6] Saved to {output_path}")
        print(f"    Measures: {len(all_measures)}")
        print(f"    Beats: {total_beats}")
        print(f"    Notes: {total_notes}")
        
        return output_path
    
    def _create_gp5(self, measures_data: List[Dict], title: str, tempo: int) -> gp.Song:
        """GP5 파일 생성"""
        song = gp.Song()
        song.title = title
        song.artist = "OmniTab (Hybrid)"
        song.tempo = tempo
        
        track = song.tracks[0]
        track.name = "Hybrid TAB"
        track.fretCount = 24
        track.channel.instrument = 25
        
        track.strings = [
            gp.GuitarString(1, 64),
            gp.GuitarString(2, 59),
            gp.GuitarString(3, 55),
            gp.GuitarString(4, 50),
            gp.GuitarString(5, 45),
            gp.GuitarString(6, 40),
        ]
        
        track.measures.clear()
        song.measureHeaders.clear()
        
        # Duration 매핑
        duration_map = {
            'whole': 1,
            'half': 2,
            'quarter': 4,
            'eighth': 8,
            'sixteenth': 16
        }
        
        for i, m_data in enumerate(measures_data):
            # 마디 헤더
            header = gp.MeasureHeader()
            header.number = i + 1
            header.start = 960 * i * 4
            header.timeSignature = gp.TimeSignature(4, gp.Duration(1))
            song.measureHeaders.append(header)
            
            # 마디
            measure = gp.Measure(track, header)
            voice = measure.voices[0]
            
            beats = m_data.get('beats', [])
            current_start = 0
            
            for b_data in beats:
                duration_str = b_data.get('duration', 'quarter')
                duration_val = duration_map.get(duration_str, 4)
                
                beat = gp.Beat(voice)
                beat.start = current_start
                beat.duration = gp.Duration(value=duration_val)
                
                notes = b_data.get('notes', [])
                used_strings = set()
                
                for n_data in notes:
                    string = n_data.get('string')
                    fret = n_data.get('fret')
                    
                    if string is None or fret is None:
                        continue
                    
                    if isinstance(fret, str):
                        if fret.upper() == 'X':
                            continue
                        try:
                            fret = int(fret)
                        except:
                            continue
                    
                    if not (1 <= string <= 6) or not (0 <= fret <= 24):
                        continue
                    
                    if string in used_strings:
                        continue
                    used_strings.add(string)
                    
                    note = gp.Note(beat)
                    note.string = string
                    note.value = fret
                    note.velocity = 95
                    note.type = gp.NoteType.normal
                    
                    beat.notes.append(note)
                
                if beat.notes:
                    beat.status = gp.BeatStatus.normal
                else:
                    beat.status = gp.BeatStatus.rest
                
                voice.beats.append(beat)
                current_start += 960 // duration_val * 4
            
            # 빈 마디면 쉼표 추가
            if not voice.beats:
                rest = gp.Beat(voice)
                rest.status = gp.BeatStatus.rest
                rest.duration = gp.Duration(value=1)
                voice.beats.append(rest)
            
            track.measures.append(measure)
        
        return song


if __name__ == "__main__":
    converter = HybridConverter()
    
    result = converter.convert(
        image_path="test_samples/images/page_1.png",
        output_path="hybrid_output.gp5",
        title="Hybrid Test",
        tempo=65
    )
    
    if result:
        print(f"\n✅ GP5 created: {result}")
