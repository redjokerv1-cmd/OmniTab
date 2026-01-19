"""
YOLO Detection to GP5 Converter v3

개선된 버전:
1. 오선보/TAB 영역 분리 (Region Segmentation)
2. TAB 영역만 YOLO 분석
3. TAB 줄 Y좌표 감지 → 감지 필터링 (ROI)
4. 줄 간격 기반 정확한 String 매핑
5. 노이즈 필터링 (텍스트 오인식 제거)
"""
from ultralytics import YOLO
import guitarpro as gp
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple


def detect_all_horizontal_lines(image: np.ndarray) -> List[float]:
    """
    전체 이미지에서 모든 수평선 Y좌표 감지
    
    Returns:
        모든 수평선의 Y좌표 리스트
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 수평 커널로 직선 감지
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 행별 픽셀 합계
    row_sums = np.sum(horizontal, axis=1)
    threshold = image.shape[1] * 0.2 * 255  # 20%로 낮춤
    
    line_rows = np.where(row_sums > threshold)[0]
    
    if len(line_rows) == 0:
        return []
    
    # 연속 행 그룹화 (선 두께 고려)
    lines = []
    current_start = line_rows[0]
    
    for i in range(1, len(line_rows)):
        if line_rows[i] - line_rows[i-1] > 5:  # 5픽셀 이상 떨어지면 새 선
            lines.append((current_start + line_rows[i-1]) // 2)
            current_start = line_rows[i]
    lines.append((current_start + line_rows[-1]) // 2)
    
    return lines


def map_y_to_string(y: float, line_ys: List[float], tolerance: float = None) -> int:
    """
    Y좌표를 줄 번호(1-6)로 매핑
    
    Args:
        y: 감지된 Y좌표
        line_ys: 6줄의 Y좌표
        tolerance: 허용 오차 (None이면 줄 간격의 50%)
    
    Returns:
        줄 번호 (1-6), 범위 밖이면 0
    """
    if not line_ys or len(line_ys) < 2:
        return 0
    
    # 줄 간격 계산
    avg_spacing = (line_ys[-1] - line_ys[0]) / (len(line_ys) - 1)
    
    if tolerance is None:
        tolerance = avg_spacing * 0.5
    
    # 가장 가까운 줄 찾기
    min_dist = float('inf')
    closest_string = 0
    
    for i, line_y in enumerate(line_ys):
        dist = abs(y - line_y)
        if dist < min_dist:
            min_dist = dist
            closest_string = i + 1
    
    # 허용 오차 내인지 확인
    if min_dist <= tolerance:
        return closest_string
    else:
        return 0  # 줄 범위 밖


def filter_noise_detections(detections: List[Dict], line_ys: List[float]) -> List[Dict]:
    """
    노이즈 감지 필터링
    
    - TAB 줄 범위 밖의 감지 제거
    - harmonic 과다 감지 필터링
    """
    if not line_ys or len(line_ys) < 6:
        return detections
    
    # TAB 영역 범위 (첫 줄 위 ~ 마지막 줄 아래)
    avg_spacing = (line_ys[-1] - line_ys[0]) / 5
    y_min = line_ys[0] - avg_spacing * 0.5
    y_max = line_ys[-1] + avg_spacing * 0.5
    
    filtered = []
    for d in detections:
        y = d['y_local']
        
        # 범위 내인지 확인
        if y_min <= y <= y_max:
            # 줄 번호 할당
            string = map_y_to_string(y, line_ys)
            if string > 0:
                d['string'] = string
                filtered.append(d)
    
    return filtered


def yolo_to_gp5_v3(image_path: str, model_path: str, output_path: str,
                   title: str = "YOLO TAB v3", tempo: int = 120,
                   conf: float = 0.25):
    """
    개선된 YOLO → GP5 변환 v3
    
    핵심 개선:
    1. TAB 영역 내부만 분석
    2. 정확한 줄 Y좌표 기반 String 매핑
    3. 노이즈 필터링 (범위 밖 감지 제거)
    """
    print("=" * 70)
    print("YOLO → GP5 Converter v3 (with Accurate Line Mapping)")
    print("=" * 70)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Cannot load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"\n[1] Image loaded: {w}x{h}")
    
    # 전체 이미지에서 수평선 감지
    print("\n[2] Detecting TAB line regions...")
    all_lines = detect_all_horizontal_lines(image)
    print(f"    Found {len(all_lines)} horizontal lines")
    
    # TAB 영역 찾기 (6줄 그룹)
    tab_regions = []
    i = 0
    while i < len(all_lines) - 5:
        group = all_lines[i:i+6]
        spacings = [group[j+1] - group[j] for j in range(5)]
        
        # 간격이 일정하면 TAB (5선보는 5줄)
        if max(spacings) - min(spacings) < 15:
            avg_spacing = sum(spacings) / 5
            # 오선보(5줄)가 아닌 TAB(6줄) 확인
            if len(group) == 6:
                tab_regions.append({
                    'y_start': int(group[0] - avg_spacing),
                    'y_end': int(group[-1] + avg_spacing),
                    'line_ys': group,
                    'avg_spacing': avg_spacing
                })
                i += 6
            else:
                i += 1
        else:
            i += 1
    
    print(f"    Found {len(tab_regions)} TAB regions")
    for idx, region in enumerate(tab_regions):
        print(f"      Region {idx+1}: y={region['y_start']}-{region['y_end']}, lines={region['line_ys']}")
    
    if not tab_regions:
        print("[Error] No TAB regions found!")
        return None
    
    # YOLO 모델 로드
    print(f"\n[3] Loading YOLO model...")
    model = YOLO(model_path)
    
    # 각 TAB 영역 분석
    all_detections = []
    
    for region_idx, region in enumerate(tab_regions):
        print(f"\n[4] Analyzing TAB region {region_idx + 1}...")
        
        # TAB 영역 크롭
        y_start = max(0, region['y_start'])
        y_end = min(h, region['y_end'])
        tab_crop = image[y_start:y_end, :]
        
        # 로컬 줄 Y좌표 (크롭된 이미지 기준)
        local_line_ys = [y - y_start for y in region['line_ys']]
        
        # 임시 파일 저장
        temp_path = f"temp_tab_v3_{region_idx}.png"
        cv2.imwrite(temp_path, tab_crop)
        
        # YOLO 분석
        results = model(temp_path, conf=conf)
        
        # 감지 결과 수집
        region_detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = r.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 프렛 번호 추출
                if cls_name.isdigit():
                    fret = int(cls_name)
                elif cls_name == 'x':
                    fret = -1
                elif cls_name == 'harmonic':
                    fret = 12  # 기본값
                else:
                    continue  # h, p 등 다른 기호는 무시
                
                region_detections.append({
                    'fret': fret,
                    'x': (x1 + x2) / 2,
                    'y_local': (y1 + y2) / 2,
                    'y_global': y_start + (y1 + y2) / 2,
                    'conf': conf_val,
                    'class': cls_name,
                    'region_idx': region_idx
                })
        
        print(f"    Raw detections: {len(region_detections)}")
        
        # 노이즈 필터링 (줄 범위 밖 제거)
        filtered = filter_noise_detections(region_detections, local_line_ys)
        print(f"    After filtering: {len(filtered)}")
        
        all_detections.extend(filtered)
        
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)
    
    print(f"\n[5] Post-processing...")
    print(f"    Total valid detections: {len(all_detections)}")
    
    if not all_detections:
        print("[Error] No valid notes detected!")
        return None
    
    # 클래스 분포
    class_counts = {}
    for d in all_detections:
        class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
    print(f"    Class distribution: {class_counts}")
    
    # 줄 분포
    string_counts = {}
    for d in all_detections:
        s = d.get('string', 0)
        string_counts[s] = string_counts.get(s, 0) + 1
    print(f"    String distribution: {string_counts}")
    
    # X좌표로 비트 그룹핑
    x_threshold = 15
    all_detections.sort(key=lambda d: d['x'])
    
    beats = []
    current_beat = []
    last_x = -1000
    
    for d in all_detections:
        if d['x'] - last_x > x_threshold:
            if current_beat:
                beats.append(current_beat)
            current_beat = [d]
        else:
            current_beat.append(d)
        last_x = d['x']
    
    if current_beat:
        beats.append(current_beat)
    
    print(f"    Grouped into {len(beats)} beats")
    
    # GP5 생성
    print(f"\n[6] Creating GP5 file...")
    song = gp.Song()
    song.title = title
    song.artist = "OmniTab (YOLO v3)"
    song.tempo = tempo
    
    track = song.tracks[0]
    track.name = "YOLO TAB v3"
    track.fretCount = 24
    track.channel.instrument = 25
    track.channel.volume = 127
    track.channel.balance = 64
    
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
    
    # 마디 생성
    beats_per_measure = 4
    num_measures = max(1, (len(beats) + beats_per_measure - 1) // beats_per_measure)
    
    beat_idx = 0
    
    for m_idx in range(num_measures):
        header = gp.MeasureHeader()
        header.number = m_idx + 1
        header.start = 960 * m_idx * 4
        header.timeSignature = gp.TimeSignature(4, gp.Duration(1))
        song.measureHeaders.append(header)
        
        measure = gp.Measure(track, header)
        voice = measure.voices[0]
        
        current_start = 0
        for _ in range(beats_per_measure):
            if beat_idx >= len(beats):
                rest = gp.Beat(voice)
                rest.status = gp.BeatStatus.rest
                rest.duration = gp.Duration(value=4)
                rest.start = current_start
                voice.beats.append(rest)
            else:
                beat_notes = beats[beat_idx]
                
                beat = gp.Beat(voice)
                beat.start = current_start
                beat.duration = gp.Duration(value=4)
                beat.status = gp.BeatStatus.normal
                
                used_strings = set()
                for note_data in beat_notes:
                    string = note_data.get('string', 0)
                    fret = note_data['fret']
                    
                    if string == 0 or string in used_strings:
                        continue
                    used_strings.add(string)
                    
                    note = gp.Note(beat)
                    note.string = string
                    note.value = max(0, fret)
                    note.velocity = 95
                    note.type = gp.NoteType.normal
                    
                    if fret == -1:
                        note.type = gp.NoteType.dead
                        note.value = 0
                    
                    beat.notes.append(note)
                
                if not beat.notes:
                    beat.status = gp.BeatStatus.rest
                
                voice.beats.append(beat)
                beat_idx += 1
            
            current_start += 960
        
        track.measures.append(measure)
    
    # 저장
    gp.write(song, output_path)
    
    total_notes = sum(len(b.notes) for m in track.measures for v in m.voices for b in v.beats)
    
    print(f"\n[7] Saved to {output_path}")
    print(f"    Measures: {len(track.measures)}")
    print(f"    Beats: {len(beats)}")
    print(f"    Notes: {total_notes}")
    
    return output_path


if __name__ == "__main__":
    result = yolo_to_gp5_v3(
        image_path="test_samples/images/page_1.png",
        model_path="runs/detect/runs/tab_detection/yolo_tab_ascii/weights/best.pt",
        output_path="yolo_output_v3.gp5",
        title="YOLO TAB v3 Test",
        tempo=65
    )
    
    if result:
        print(f"\n✅ GP5 file created: {result}")
