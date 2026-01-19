"""
YOLO Detection to GP5 Converter
Y좌표로 줄 번호, X좌표로 비트 순서 결정
"""
from ultralytics import YOLO
import guitarpro as gp
from pathlib import Path


# 튜닝 노트 to MIDI pitch
TUNING_TO_MIDI = {
    'E': 64, 'B': 59, 'G': 55, 'D': 50, 'A': 45, 'E2': 40,
    'C': 60, 'D#': 63, 'F': 65, 'F#': 66, 'G#': 68, 'A#': 70
}


def yolo_to_gp5(image_path: str, model_path: str, output_path: str,
                title: str = "YOLO TAB", tempo: int = 120,
                conf: float = 0.25):
    """
    YOLO 감지 결과를 GP5 파일로 변환
    """
    print(f"[YOLO→GP5] Loading model...")
    model = YOLO(model_path)
    
    print(f"[YOLO→GP5] Detecting notes in {image_path}...")
    results = model(image_path, conf=conf)
    
    # 감지 결과 파싱
    detections = []
    img_height = None
    
    for r in results:
        img_height = r.orig_shape[0]
        
        for box in r.boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = r.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            # 프렛 번호 추출
            if cls_name.isdigit():
                fret = int(cls_name)
            elif cls_name == 'x':
                fret = -1  # 뮤트
            elif cls_name == 'harmonic':
                fret = 12  # 하모닉은 보통 12프렛
            else:
                continue
            
            detections.append({
                'fret': fret,
                'x': x_center,
                'y': y_center,
                'conf': conf_val,
                'class': cls_name
            })
    
    print(f"[YOLO→GP5] Detected {len(detections)} notes")
    
    if not detections:
        print("[YOLO→GP5] No notes detected!")
        return
    
    # Y좌표로 줄 번호 결정
    y_values = [d['y'] for d in detections]
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    
    # 6줄로 나누기
    num_strings = 6
    line_height = y_range / (num_strings - 1) if y_range > 0 else 50
    
    # 각 감지에 줄 번호 할당
    for d in detections:
        relative_y = d['y'] - y_min
        string_num = round(relative_y / line_height) + 1
        string_num = max(1, min(6, string_num))
        d['string'] = string_num
    
    # X좌표로 비트 그룹핑
    x_threshold = 20
    detections.sort(key=lambda d: d['x'])
    
    beats = []
    current_beat = []
    last_x = -1000
    
    for d in detections:
        if d['x'] - last_x > x_threshold:
            if current_beat:
                beats.append(current_beat)
            current_beat = [d]
        else:
            current_beat.append(d)
        last_x = d['x']
    
    if current_beat:
        beats.append(current_beat)
    
    print(f"[YOLO→GP5] Grouped into {len(beats)} beats")
    
    # GP5 파일 생성 (올바른 방식)
    song = gp.Song()
    song.title = title
    song.artist = "OmniTab (YOLO)"
    song.tempo = tempo
    
    # 기본 트랙 사용 (중요!)
    track = song.tracks[0]
    track.name = "YOLO TAB"
    track.fretCount = 24
    
    # 채널 설정
    track.channel.instrument = 25  # Acoustic Guitar
    track.channel.volume = 127
    track.channel.balance = 64
    
    # 스트링 설정 (표준 튜닝)
    track.strings = [
        gp.GuitarString(1, 64),  # E4
        gp.GuitarString(2, 59),  # B3
        gp.GuitarString(3, 55),  # G3
        gp.GuitarString(4, 50),  # D3
        gp.GuitarString(5, 45),  # A2
        gp.GuitarString(6, 40),  # E2
    ]
    
    # 기존 measures 초기화
    track.measures.clear()
    song.measureHeaders.clear()
    
    # 마디 생성 (4비트씩 그룹핑)
    beats_per_measure = 4
    num_measures = max(1, (len(beats) + beats_per_measure - 1) // beats_per_measure)
    
    beat_idx = 0
    
    for m_idx in range(num_measures):
        # 마디 헤더 생성
        header = gp.MeasureHeader()
        header.number = m_idx + 1
        header.start = 960 * m_idx * 4
        header.timeSignature = gp.TimeSignature(4, gp.Duration(1))
        song.measureHeaders.append(header)
        
        # 마디 생성
        measure = gp.Measure(track, header)
        voice = measure.voices[0]
        
        # 이 마디에 들어갈 비트들
        current_start = 0
        for _ in range(beats_per_measure):
            if beat_idx >= len(beats):
                # 나머지는 쉼표
                rest = gp.Beat(voice)
                rest.status = gp.BeatStatus.rest
                rest.duration = gp.Duration(value=4)
                rest.start = current_start
                voice.beats.append(rest)
            else:
                beat_notes = beats[beat_idx]
                
                # 비트 생성
                beat = gp.Beat(voice)
                beat.start = current_start
                beat.duration = gp.Duration(value=4)  # 4분음표
                beat.status = gp.BeatStatus.normal
                
                # 노트 추가
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
                    
                    # 뮤트 처리
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
    
    print(f"[YOLO→GP5] Saved to {output_path}")
    print(f"[YOLO→GP5] Total measures: {len(track.measures)}")
    print(f"[YOLO→GP5] Total beats: {len(beats)}")
    
    total_notes = sum(len(b) for b in beats)
    print(f"[YOLO→GP5] Total notes: {total_notes}")
    
    return output_path


if __name__ == "__main__":
    result = yolo_to_gp5(
        image_path="test_samples/images/page_1.png",
        model_path="runs/detect/runs/tab_detection/yolo_tab_ascii/weights/best.pt",
        output_path="yolo_output.gp5",
        title="YOLO TAB Test",
        tempo=65
    )
    
    if result:
        print(f"\n✅ GP5 file created: {result}")
        print("Open with Guitar Pro to verify!")
