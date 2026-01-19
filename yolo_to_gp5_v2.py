"""
YOLO Detection to GP5 Converter v2

개선된 버전:
1. 오선보/TAB 영역 분리
2. TAB 영역만 YOLO 분석
3. 오선보에서 오인된 음표 제거
"""
from ultralytics import YOLO
import guitarpro as gp
from pathlib import Path
import cv2
import numpy as np

from omnitab.tab_ocr.region_segmenter import RegionSegmenter


def yolo_to_gp5_v2(image_path: str, model_path: str, output_path: str,
                   title: str = "YOLO TAB v2", tempo: int = 120,
                   conf: float = 0.25):
    """
    개선된 YOLO → GP5 변환
    
    1단계: 영역 분리 (오선보 vs TAB)
    2단계: TAB 영역만 YOLO 분석
    3단계: GP5 생성
    """
    print("=" * 60)
    print("YOLO → GP5 Converter v2 (with Region Segmentation)")
    print("=" * 60)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Cannot load image: {image_path}")
        return None
    
    print(f"\n[1] Region Segmentation...")
    segmenter = RegionSegmenter()
    staff_regions, tab_regions = segmenter.segment(image)
    
    print(f"    Found {len(staff_regions)} staff regions")
    print(f"    Found {len(tab_regions)} TAB regions")
    
    if not tab_regions:
        print("[Warning] No TAB regions found, analyzing full image")
        tab_regions = [type('Region', (), {
            'y_start': 0, 'y_end': image.shape[0],
            'x_start': 0, 'x_end': image.shape[1],
            'crop': lambda self, img: img
        })()]
    
    # YOLO 모델 로드
    print(f"\n[2] Loading YOLO model...")
    model = YOLO(model_path)
    
    # 각 TAB 영역 분석
    all_detections = []
    
    for i, region in enumerate(tab_regions):
        print(f"\n[3] Analyzing TAB region {i+1}...")
        
        # TAB 영역 크롭
        tab_crop = image[region.y_start:region.y_end, :]
        
        # 임시 파일로 저장 (YOLO는 파일 경로 필요)
        temp_path = f"temp_tab_region_{i}.png"
        cv2.imwrite(temp_path, tab_crop)
        
        # YOLO 분석
        results = model(temp_path, conf=conf)
        
        # 감지 결과 수집
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
                    # 오선보 음표 오인 가능성 - 더 엄격한 필터링
                    # harmonic은 보통 특정 프렛(5, 7, 12) 근처에서만 나타남
                    fret = 12
                else:
                    continue
                
                # 전역 좌표로 변환
                y_center_global = region.y_start + (y1 + y2) / 2
                x_center = (x1 + x2) / 2
                
                all_detections.append({
                    'fret': fret,
                    'x': x_center,
                    'y': y_center_global,
                    'y_local': (y1 + y2) / 2,  # 영역 내 상대 좌표
                    'region_idx': i,
                    'region_height': region.y_end - region.y_start,
                    'conf': conf_val,
                    'class': cls_name
                })
        
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)
        
        print(f"    Detected {len([d for d in all_detections if d['region_idx'] == i])} notes")
    
    print(f"\n[4] Post-processing...")
    print(f"    Total detections: {len(all_detections)}")
    
    # harmonic 필터링 (너무 많으면 오인)
    harmonic_count = len([d for d in all_detections if d['class'] == 'harmonic'])
    if harmonic_count > 5:
        print(f"    [Filter] Too many harmonics ({harmonic_count}), removing...")
        all_detections = [d for d in all_detections if d['class'] != 'harmonic']
        print(f"    After filter: {len(all_detections)} notes")
    
    if not all_detections:
        print("[Error] No notes detected!")
        return None
    
    # 각 영역별로 줄 번호 결정
    for d in all_detections:
        # 해당 영역의 6줄 중 어디인지 계산
        region_height = d['region_height']
        y_local = d['y_local']
        
        # 6줄이 균등 배치되어 있다고 가정
        # 상단 여백 ~15%, 하단 여백 ~15%
        usable_height = region_height * 0.7
        y_offset = region_height * 0.15
        
        relative_y = (y_local - y_offset) / usable_height
        string_num = int(relative_y * 5) + 1
        string_num = max(1, min(6, string_num))
        
        d['string'] = string_num
    
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
    
    # 클래스 분포 출력
    class_counts = {}
    for d in all_detections:
        class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
    
    print(f"    Class distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      {cls}: {count}")
    
    # GP5 생성
    print(f"\n[5] Creating GP5 file...")
    song = gp.Song()
    song.title = title
    song.artist = "OmniTab (YOLO v2)"
    song.tempo = tempo
    
    track = song.tracks[0]
    track.name = "YOLO TAB v2"
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
                    string = note_data['string']
                    fret = note_data['fret']
                    
                    if string in used_strings:
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
    
    print(f"\n[6] Saved to {output_path}")
    print(f"    Measures: {len(track.measures)}")
    print(f"    Beats: {len(beats)}")
    total_notes = sum(len(b) for b in beats)
    print(f"    Notes: {total_notes}")
    
    return output_path


if __name__ == "__main__":
    result = yolo_to_gp5_v2(
        image_path="test_samples/images/page_1.png",
        model_path="runs/detect/runs/tab_detection/yolo_tab_ascii/weights/best.pt",
        output_path="yolo_output_v2.gp5",
        title="YOLO TAB v2 Test",
        tempo=65
    )
    
    if result:
        print(f"\n✅ GP5 file created: {result}")
