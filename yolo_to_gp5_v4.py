"""
YOLO Detection to GP5 Converter v4

핵심 개선:
1. 모든 TAB 시스템(System) 감지 - 페이지 전체 스캔
2. Y축 클러스터링 - 노트를 해당 시스템에 할당
3. 시간축 재정렬 - System 순서 → X좌표 순서 (Z-order)
4. 신뢰도 필터링 - 낮은 confidence 노이즈 제거
5. ROI 필터링 - 타브선 범위 밖 노이즈 제거
"""
from ultralytics import YOLO
import guitarpro as gp
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TabSystem:
    """하나의 TAB 시스템 (6줄 묶음)"""
    index: int
    y_min: float
    y_max: float
    line_ys: List[float]  # 6줄의 Y좌표
    avg_spacing: float


@dataclass
class NoteEvent:
    """할당된 노트 이벤트"""
    system_idx: int
    string: int  # 1-6
    fret: int
    x: float
    confidence: float


def detect_all_tab_systems(image: np.ndarray) -> List[TabSystem]:
    """
    페이지 전체에서 모든 TAB 시스템(6줄 묶음) 감지
    
    Returns:
        TabSystem 리스트 (위에서 아래 순서)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 수평선 감지
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 행별 픽셀 합계
    row_sums = np.sum(horizontal, axis=1)
    threshold = image.shape[1] * 0.15 * 255  # 15%로 더 낮춤
    
    line_rows = np.where(row_sums > threshold)[0]
    
    if len(line_rows) == 0:
        return []
    
    # 연속 행 그룹화
    lines = []
    current_start = line_rows[0]
    
    for i in range(1, len(line_rows)):
        if line_rows[i] - line_rows[i-1] > 5:
            lines.append((current_start + line_rows[i-1]) // 2)
            current_start = line_rows[i]
    lines.append((current_start + line_rows[-1]) // 2)
    
    # 모든 6줄 그룹 찾기
    systems = []
    i = 0
    system_idx = 0
    
    while i < len(lines) - 5:
        group = lines[i:i+6]
        spacings = [group[j+1] - group[j] for j in range(5)]
        avg_spacing = sum(spacings) / 5
        
        # 간격이 일정하면 TAB 시스템
        if max(spacings) - min(spacings) < 15 and 10 < avg_spacing < 40:
            systems.append(TabSystem(
                index=system_idx,
                y_min=float(group[0] - avg_spacing * 0.5),
                y_max=float(group[-1] + avg_spacing * 0.5),
                line_ys=[float(y) for y in group],
                avg_spacing=avg_spacing
            ))
            system_idx += 1
            i += 6  # 다음 그룹으로
        else:
            i += 1
    
    return systems


def find_closest_string(y: float, line_ys: List[float], margin: float) -> int:
    """
    Y좌표를 줄 번호(1-6)로 매핑
    
    Returns:
        줄 번호 (1-6), 범위 밖이면 0
    """
    if not line_ys:
        return 0
    
    min_dist = float('inf')
    closest = 0
    
    for i, line_y in enumerate(line_ys):
        dist = abs(y - line_y)
        if dist < min_dist:
            min_dist = dist
            closest = i + 1
    
    # 허용 오차 확인
    if min_dist <= margin:
        return closest
    return 0


def assign_notes_to_systems(detections: List[Dict], systems: List[TabSystem],
                           conf_threshold: float = 0.4) -> List[NoteEvent]:
    """
    YOLO 감지 결과를 적절한 TAB 시스템에 할당
    
    핵심 로직:
    1. 신뢰도 필터링
    2. 시스템 범위 확인
    3. 가장 가까운 줄에 할당
    """
    notes = []
    ignored_count = 0
    
    for det in detections:
        # 신뢰도 필터링
        if det['confidence'] < conf_threshold:
            ignored_count += 1
            continue
        
        # 숫자 클래스만 처리
        cls_name = det['class_name']
        if cls_name.isdigit():
            fret = int(cls_name)
        elif cls_name == 'x':
            fret = -1
        else:
            continue  # harmonic 등 다른 것은 제외
        
        y = det['y_abs']
        x = det['x_abs']
        
        # 해당 시스템 찾기
        assigned = False
        for system in systems:
            margin = system.avg_spacing * 1.0  # margin 확대
            
            if system.y_min - margin <= y <= system.y_max + margin:
                # 가장 가까운 줄 찾기
                string = find_closest_string(y, system.line_ys, margin)
                
                if string > 0:
                    notes.append(NoteEvent(
                        system_idx=system.index,
                        string=string,
                        fret=fret,
                        x=x,
                        confidence=det['confidence']
                    ))
                    assigned = True
                    break
        
        if not assigned:
            ignored_count += 1
    
    return notes, ignored_count


def sort_notes_z_order(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    노트를 Z-order로 정렬 (위→아래, 좌→우)
    
    악보 읽기 순서: System 1 (좌→우) → System 2 (좌→우) → ...
    """
    return sorted(notes, key=lambda n: (n.system_idx, n.x))


def group_into_beats(notes: List[NoteEvent], x_threshold: float = 15) -> List[List[NoteEvent]]:
    """
    동시에 연주되는 노트를 비트로 그룹화
    """
    if not notes:
        return []
    
    beats = []
    current_beat = [notes[0]]
    
    for i in range(1, len(notes)):
        # 같은 시스템이고 X좌표가 가까우면 같은 비트
        if (notes[i].system_idx == current_beat[-1].system_idx and
            abs(notes[i].x - current_beat[-1].x) < x_threshold):
            current_beat.append(notes[i])
        else:
            beats.append(current_beat)
            current_beat = [notes[i]]
    
    beats.append(current_beat)
    return beats


def yolo_to_gp5_v4(image_path: str, model_path: str, output_path: str,
                   title: str = "YOLO TAB v4", tempo: int = 120,
                   conf: float = 0.3):
    """
    YOLO → GP5 변환 v4
    
    핵심 개선:
    - 모든 TAB 시스템 감지
    - Z-order 시간축 정렬
    - 신뢰도 기반 노이즈 필터링
    """
    print("=" * 70)
    print("YOLO → GP5 Converter v4 (Multi-System + Z-Order)")
    print("=" * 70)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Cannot load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"\n[1] Image: {w}x{h}")
    
    # 모든 TAB 시스템 감지
    print("\n[2] Detecting all TAB systems...")
    systems = detect_all_tab_systems(image)
    print(f"    Found {len(systems)} TAB systems:")
    
    for sys in systems:
        print(f"      System {sys.index + 1}: Y={sys.y_min:.0f}-{sys.y_max:.0f}, spacing={sys.avg_spacing:.1f}")
    
    if not systems:
        print("[Error] No TAB systems found!")
        return None
    
    # YOLO 분석
    print(f"\n[3] Running YOLO detection...")
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    
    # 원시 감지 수집
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = r.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                'class_name': cls_name,
                'confidence': conf_val,
                'x_abs': (x1 + x2) / 2,
                'y_abs': (y1 + y2) / 2
            })
    
    print(f"    Raw detections: {len(detections)}")
    
    # 클래스 분포
    class_dist = {}
    for d in detections:
        class_dist[d['class_name']] = class_dist.get(d['class_name'], 0) + 1
    print(f"    Classes: {class_dist}")
    
    # 시스템에 노트 할당
    print(f"\n[4] Assigning notes to systems (conf >= 0.25)...")
    notes, ignored = assign_notes_to_systems(detections, systems, conf_threshold=0.25)
    print(f"    Valid notes: {len(notes)}")
    print(f"    Ignored (noise/low conf): {ignored}")
    
    if not notes:
        print("[Error] No valid notes after filtering!")
        return None
    
    # 시스템별 분포
    sys_dist = {}
    for n in notes:
        sys_dist[n.system_idx] = sys_dist.get(n.system_idx, 0) + 1
    print(f"    Notes per system: {sys_dist}")
    
    # 줄별 분포
    string_dist = {}
    for n in notes:
        string_dist[n.string] = string_dist.get(n.string, 0) + 1
    print(f"    Notes per string: {string_dist}")
    
    # Z-order 정렬
    print(f"\n[5] Sorting notes in Z-order (reading order)...")
    sorted_notes = sort_notes_z_order(notes)
    
    # 비트 그룹화
    beats = group_into_beats(sorted_notes)
    print(f"    Grouped into {len(beats)} beats")
    
    # GP5 생성
    print(f"\n[6] Creating GP5 file...")
    song = gp.Song()
    song.title = title
    song.artist = "OmniTab (YOLO v4)"
    song.tempo = tempo
    
    track = song.tracks[0]
    track.name = "YOLO TAB v4"
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
                for note in beat_notes:
                    if note.string in used_strings:
                        continue
                    used_strings.add(note.string)
                    
                    gp_note = gp.Note(beat)
                    gp_note.string = note.string
                    gp_note.value = max(0, note.fret)
                    gp_note.velocity = 95
                    gp_note.type = gp.NoteType.normal
                    
                    if note.fret == -1:
                        gp_note.type = gp.NoteType.dead
                        gp_note.value = 0
                    
                    beat.notes.append(gp_note)
                
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
    print(f"    TAB Systems: {len(systems)}")
    print(f"    Measures: {len(track.measures)}")
    print(f"    Beats: {len(beats)}")
    print(f"    Notes: {total_notes}")
    
    return output_path


if __name__ == "__main__":
    result = yolo_to_gp5_v4(
        image_path="test_samples/images/page_1.png",
        model_path="runs/detect/runs/tab_detection/yolo_tab_ascii/weights/best.pt",
        output_path="yolo_output_v4.gp5",
        title="YOLO TAB v4 Test",
        tempo=65
    )
    
    if result:
        print(f"\n✅ GP5 file created: {result}")
