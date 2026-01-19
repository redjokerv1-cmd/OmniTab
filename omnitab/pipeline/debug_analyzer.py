"""
Debug Analyzer - 전체 파이프라인 상세 분석 및 JSON 출력

모든 중간 단계를 JSON으로 내보내서 디버깅 가능하게 함:
1. YOLO 원시 출력
2. TAB 줄 좌표
3. 그룹화된 이벤트
4. 오선보 분석 (추가)
5. 마디 구조
6. GP5 변환 로그
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from ultralytics import YOLO


@dataclass
class Detection:
    """YOLO 감지 결과"""
    class_name: str
    class_id: int
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    x_abs: float
    y_abs: float


@dataclass
class TabLine:
    """TAB 줄 정보"""
    line_index: int
    y_position: float
    string_number: int


@dataclass 
class FretEvent:
    """프렛 이벤트"""
    string: int
    fret: int
    confidence: float
    x: float
    y: float


@dataclass
class GroupedEvent:
    """그룹화된 이벤트 (동시음)"""
    time_x: float
    frets: List[Dict]
    techniques: List[str]
    assigned_duration: str


@dataclass
class StaffNote:
    """오선보 음표 (리듬 정보)"""
    x: float
    y: float
    pitch_line_index: int
    note_type: str  # whole/half/quarter/eighth/sixteenth
    dotted: bool
    beam_group_id: Optional[int]
    duration_beats: float


@dataclass
class Measure:
    """마디 정보"""
    index: int
    x_start: float
    x_end: float
    time_signature: str
    actual_beat_sum: float


class DebugAnalyzer:
    """
    전체 파이프라인 분석 및 디버깅
    """
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.debug_data = {}
    
    def analyze(self, image_path: str, output_dir: str = "debug_output") -> Dict:
        """전체 분석 실행"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        h, w = image.shape[:2]
        
        print("=" * 70)
        print("DEBUG ANALYZER - Full Pipeline Analysis")
        print("=" * 70)
        
        # 1. YOLO 원시 출력
        print("\n[1/6] YOLO Detection...")
        detections = self._run_yolo(image_path, w, h)
        self.debug_data['yolo_raw'] = {
            'image_path': str(image_path),
            'image_width': w,
            'image_height': h,
            'detection_count': len(detections),
            'detections': [asdict(d) for d in detections]
        }
        
        # 2. TAB 줄 감지
        print("[2/6] TAB Line Detection...")
        tab_lines = self._detect_tab_lines(image)
        self.debug_data['tab_lines'] = {
            'lines': [asdict(l) for l in tab_lines]
        }
        
        # 3. 마디선 감지
        print("[3/6] Barline Detection...")
        barlines, time_signature = self._detect_barlines(image)
        measures = self._create_measures(barlines, w, time_signature)
        self.debug_data['measures'] = {
            'barline_x_positions': barlines,
            'time_signature': time_signature,
            'measures': [asdict(m) for m in measures]
        }
        
        # 4. 프렛 → 줄 매핑 및 그룹화
        print("[4/6] Fret-to-String Mapping & Grouping...")
        fret_events = self._map_detections_to_strings(detections, tab_lines)
        grouped_events = self._group_simultaneous_frets(fret_events)
        self.debug_data['grouped_events'] = {
            'total_fret_events': len(fret_events),
            'grouped_event_count': len(grouped_events),
            'events': [asdict(e) for e in grouped_events]
        }
        
        # 5. 오선보 분석 (리듬)
        print("[5/6] Staff Rhythm Analysis...")
        staff_notes = self._analyze_staff_rhythm(image, grouped_events)
        self.debug_data['staff_notes'] = {
            'note_count': len(staff_notes),
            'notes': [asdict(n) for n in staff_notes]
        }
        
        # 6. 리듬 추정 (x 간격 기반)
        print("[6/6] Duration Estimation...")
        self._estimate_durations(grouped_events)
        self.debug_data['duration_estimation'] = {
            'method': 'x_spacing',
            'events_with_duration': [asdict(e) for e in grouped_events]
        }
        
        # JSON 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"debug_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.debug_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Saved] {json_path}")
        
        # 요약 출력
        self._print_summary()
        
        return self.debug_data
    
    def _run_yolo(self, image_path: str, img_w: int, img_h: int) -> List[Detection]:
        """YOLO 실행 및 원시 결과 수집"""
        results = self.model(image_path, conf=0.25)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = r.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                detections.append(Detection(
                    class_name=cls_name,
                    class_id=cls,
                    confidence=conf,
                    x_center=x_center / img_w,  # normalized
                    y_center=y_center / img_h,
                    width=width / img_w,
                    height=height / img_h,
                    x_abs=x_center,
                    y_abs=y_center
                ))
        
        print(f"    Found {len(detections)} detections")
        
        # 클래스 분포
        class_dist = {}
        for d in detections:
            class_dist[d.class_name] = class_dist.get(d.class_name, 0) + 1
        print(f"    Classes: {class_dist}")
        
        return detections
    
    def _detect_tab_lines(self, image: np.ndarray) -> List[TabLine]:
        """TAB 6줄의 Y좌표 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 수평 커널
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 행별 합계
        row_sums = np.sum(horizontal, axis=1)
        threshold = image.shape[1] * 0.2 * 255
        
        # 선 위치 찾기
        line_rows = np.where(row_sums > threshold)[0]
        
        if len(line_rows) == 0:
            print("    [Warning] No horizontal lines detected")
            return []
        
        # 그룹화
        lines = []
        current_start = line_rows[0]
        
        for i in range(1, len(line_rows)):
            if line_rows[i] - line_rows[i-1] > 5:
                lines.append((current_start + line_rows[i-1]) // 2)
                current_start = line_rows[i]
        lines.append((current_start + line_rows[-1]) // 2)
        
        # 6줄 그룹 찾기 (TAB)
        tab_lines = []
        for i in range(len(lines) - 5):
            # 연속 6줄 확인
            group = lines[i:i+6]
            spacings = [group[j+1] - group[j] for j in range(5)]
            
            # 간격이 일정하면 TAB
            if max(spacings) - min(spacings) < 10:
                for idx, y in enumerate(group):
                    tab_lines.append(TabLine(
                        line_index=idx,
                        y_position=float(y),
                        string_number=idx + 1
                    ))
                break
        
        print(f"    Found {len(tab_lines)} TAB lines")
        if tab_lines:
            print(f"    Y positions: {[l.y_position for l in tab_lines]}")
        
        return tab_lines
    
    def _detect_barlines(self, image: np.ndarray) -> Tuple[List[float], str]:
        """마디선(세로선) 감지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 수직 커널
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 열별 합계
        col_sums = np.sum(vertical, axis=0)
        threshold = image.shape[0] * 0.1 * 255
        
        barline_cols = np.where(col_sums > threshold)[0]
        
        # 그룹화
        barlines = []
        if len(barline_cols) > 0:
            current_start = barline_cols[0]
            for i in range(1, len(barline_cols)):
                if barline_cols[i] - barline_cols[i-1] > 10:
                    barlines.append((current_start + barline_cols[i-1]) / 2)
                    current_start = barline_cols[i]
            barlines.append((current_start + barline_cols[-1]) / 2)
        
        print(f"    Found {len(barlines)} barlines at x: {barlines[:5]}...")
        
        # 박자표 (TODO: OCR로 추출)
        time_signature = "4/4"  # 기본값
        
        return barlines, time_signature
    
    def _create_measures(self, barlines: List[float], img_width: int, 
                        time_sig: str) -> List[Measure]:
        """마디 구조 생성"""
        if not barlines:
            return [Measure(0, 0, img_width, time_sig, 0)]
        
        measures = []
        prev_x = 0
        
        for i, x in enumerate(barlines):
            measures.append(Measure(
                index=i,
                x_start=prev_x,
                x_end=x,
                time_signature=time_sig,
                actual_beat_sum=0  # 나중에 계산
            ))
            prev_x = x
        
        # 마지막 마디
        if prev_x < img_width:
            measures.append(Measure(
                index=len(measures),
                x_start=prev_x,
                x_end=img_width,
                time_signature=time_sig,
                actual_beat_sum=0
            ))
        
        return measures
    
    def _map_detections_to_strings(self, detections: List[Detection], 
                                   tab_lines: List[TabLine]) -> List[FretEvent]:
        """YOLO 감지 → 줄 번호 매핑"""
        if not tab_lines:
            print("    [Warning] No TAB lines, cannot map strings")
            return []
        
        line_ys = [l.y_position for l in tab_lines]
        
        fret_events = []
        for d in detections:
            # 숫자만 처리
            if not d.class_name.isdigit():
                continue
            
            fret = int(d.class_name)
            
            # 가장 가까운 줄 찾기
            min_dist = float('inf')
            closest_string = 1
            
            for i, y in enumerate(line_ys):
                dist = abs(d.y_abs - y)
                if dist < min_dist:
                    min_dist = dist
                    closest_string = i + 1
            
            fret_events.append(FretEvent(
                string=closest_string,
                fret=fret,
                confidence=d.confidence,
                x=d.x_abs,
                y=d.y_abs
            ))
        
        print(f"    Mapped {len(fret_events)} fret events to strings")
        return fret_events
    
    def _group_simultaneous_frets(self, fret_events: List[FretEvent], 
                                  x_tolerance: float = 15) -> List[GroupedEvent]:
        """동시음 그룹화"""
        if not fret_events:
            return []
        
        # X좌표 정렬
        sorted_events = sorted(fret_events, key=lambda e: e.x)
        
        groups = []
        current_group = [sorted_events[0]]
        
        for i in range(1, len(sorted_events)):
            if sorted_events[i].x - current_group[-1].x < x_tolerance:
                current_group.append(sorted_events[i])
            else:
                # 그룹 완료
                avg_x = sum(e.x for e in current_group) / len(current_group)
                groups.append(GroupedEvent(
                    time_x=avg_x,
                    frets=[{'string': e.string, 'fret': e.fret, 'confidence': e.confidence} 
                           for e in current_group],
                    techniques=[],
                    assigned_duration='quarter'  # 기본값
                ))
                current_group = [sorted_events[i]]
        
        # 마지막 그룹
        if current_group:
            avg_x = sum(e.x for e in current_group) / len(current_group)
            groups.append(GroupedEvent(
                time_x=avg_x,
                frets=[{'string': e.string, 'fret': e.fret, 'confidence': e.confidence} 
                       for e in current_group],
                techniques=[],
                assigned_duration='quarter'
            ))
        
        print(f"    Grouped into {len(groups)} simultaneous events")
        
        # 화음 크기 분포
        chord_sizes = {}
        for g in groups:
            size = len(g.frets)
            chord_sizes[size] = chord_sizes.get(size, 0) + 1
        print(f"    Chord sizes: {chord_sizes}")
        
        return groups
    
    def _analyze_staff_rhythm(self, image: np.ndarray, 
                              grouped_events: List[GroupedEvent]) -> List[StaffNote]:
        """
        오선보에서 리듬 정보 추출
        
        TODO: 실제 음표 감지 구현
        현재는 x 간격 기반 추정만 함
        """
        print("    [Info] Staff rhythm analysis not fully implemented")
        print("    Using x-spacing estimation instead")
        
        # 현재는 빈 리스트 반환
        # 향후: 음표 머리 감지 → stem/beam 분석 → 리듬 결정
        return []
    
    def _estimate_durations(self, grouped_events: List[GroupedEvent]):
        """X 간격 기반 리듬 추정"""
        if len(grouped_events) < 2:
            return
        
        # 평균 간격 계산
        gaps = []
        for i in range(len(grouped_events) - 1):
            gap = grouped_events[i+1].time_x - grouped_events[i].time_x
            gaps.append(gap)
        
        if not gaps:
            return
        
        avg_gap = sum(gaps) / len(gaps)
        
        for i, event in enumerate(grouped_events[:-1]):
            gap = grouped_events[i+1].time_x - event.time_x
            
            # 간격 비율로 duration 추정
            ratio = gap / avg_gap
            
            if ratio < 0.6:
                event.assigned_duration = 'sixteenth'
            elif ratio < 0.85:
                event.assigned_duration = 'eighth'
            elif ratio < 1.5:
                event.assigned_duration = 'quarter'
            else:
                event.assigned_duration = 'half'
        
        # 마지막 음표
        grouped_events[-1].assigned_duration = 'quarter'
        
        # duration 분포
        duration_dist = {}
        for e in grouped_events:
            duration_dist[e.assigned_duration] = duration_dist.get(e.assigned_duration, 0) + 1
        print(f"    Duration distribution: {duration_dist}")
    
    def _print_summary(self):
        """분석 결과 요약"""
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        yolo = self.debug_data.get('yolo_raw', {})
        print(f"\n[YOLO Detections]")
        print(f"  Total: {yolo.get('detection_count', 0)}")
        
        tab = self.debug_data.get('tab_lines', {})
        print(f"\n[TAB Lines]")
        print(f"  Lines: {len(tab.get('lines', []))}")
        
        measures = self.debug_data.get('measures', {})
        print(f"\n[Measures]")
        print(f"  Barlines: {len(measures.get('barline_x_positions', []))}")
        print(f"  Measures: {len(measures.get('measures', []))}")
        print(f"  Time Signature: {measures.get('time_signature', 'N/A')}")
        
        events = self.debug_data.get('grouped_events', {})
        print(f"\n[Grouped Events]")
        print(f"  Total fret events: {events.get('total_fret_events', 0)}")
        print(f"  Grouped (chords): {events.get('grouped_event_count', 0)}")
        
        duration = self.debug_data.get('duration_estimation', {})
        print(f"\n[Duration Estimation]")
        print(f"  Method: {duration.get('method', 'N/A')}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    model_path = "runs/detect/runs/tab_detection/yolo_tab_ascii/weights/best.pt"
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_samples/images/page_1.png"
    
    analyzer = DebugAnalyzer(model_path)
    analyzer.analyze(image_path)
