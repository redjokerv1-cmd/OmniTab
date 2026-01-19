"""
TAB Region Segmenter

이미지에서 오선보(Staff)와 TAB 영역을 분리하는 모듈
YOLO가 오선보의 음표를 프렛으로 오인하는 문제 해결

구조:
1. 오선보: 5개의 평행선 + 음표 머리
2. TAB: 6개의 평행선 + 숫자

분리 방법:
- 수평선 감지 → 5줄 그룹 vs 6줄 그룹 분류
- 또는 "TAB" 레이블 위치 기준으로 분리
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Region:
    """감지된 영역"""
    type: str  # 'staff' or 'tab'
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    num_lines: int
    
    def crop(self, image: np.ndarray) -> np.ndarray:
        """이 영역으로 이미지 크롭"""
        return image[self.y_start:self.y_end, self.x_start:self.x_end]


class RegionSegmenter:
    """
    오선보/TAB 영역 분리기
    
    방법:
    1. 수평선 감지 (Hough Transform)
    2. 선 그룹 클러스터링
    3. 5줄 = 오선보, 6줄 = TAB로 분류
    """
    
    def __init__(self, min_line_length: int = 100, line_gap_threshold: int = 30):
        self.min_line_length = min_line_length
        self.line_gap_threshold = line_gap_threshold
    
    def detect_horizontal_lines(self, image: np.ndarray) -> List[int]:
        """수평선의 Y좌표 감지"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 이진화
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 수평 커널로 모폴로지 연산
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 각 행의 픽셀 합계 계산
        row_sums = np.sum(horizontal_lines, axis=1)
        
        # 임계값 이상인 행 = 수평선
        threshold = image.shape[1] * 0.3 * 255  # 30% 이상 채워진 행
        line_rows = np.where(row_sums > threshold)[0]
        
        # 연속된 행 그룹화 (선 두께 고려)
        if len(line_rows) == 0:
            return []
        
        lines = []
        current_line_start = line_rows[0]
        
        for i in range(1, len(line_rows)):
            if line_rows[i] - line_rows[i-1] > 3:  # 3픽셀 이상 떨어지면 새 선
                lines.append((current_line_start + line_rows[i-1]) // 2)
                current_line_start = line_rows[i]
        
        lines.append((current_line_start + line_rows[-1]) // 2)
        
        return lines
    
    def group_lines(self, lines: List[int]) -> List[List[int]]:
        """
        인접한 선들을 그룹으로 묶기
        
        오선보: 5줄, 일정한 간격
        TAB: 6줄, 일정한 간격
        """
        if len(lines) < 5:
            return []
        
        groups = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            gap = lines[i] - lines[i-1]
            
            # 같은 그룹의 선 간격 (보통 15-30픽셀)
            if gap < self.line_gap_threshold:
                current_group.append(lines[i])
            else:
                if len(current_group) >= 5:
                    groups.append(current_group)
                current_group = [lines[i]]
        
        if len(current_group) >= 5:
            groups.append(current_group)
        
        return groups
    
    def classify_groups(self, groups: List[List[int]], image_width: int) -> List[Region]:
        """
        그룹을 오선보/TAB으로 분류
        
        5줄 = 오선보
        6줄 = TAB
        """
        regions = []
        
        for group in groups:
            num_lines = len(group)
            y_start = max(0, group[0] - 20)
            y_end = group[-1] + 20
            
            if num_lines == 5:
                region_type = 'staff'
            elif num_lines == 6:
                region_type = 'tab'
            else:
                # 5줄도 6줄도 아니면 가장 가까운 것으로
                region_type = 'tab' if num_lines > 5 else 'staff'
            
            regions.append(Region(
                type=region_type,
                y_start=y_start,
                y_end=y_end,
                x_start=0,
                x_end=image_width,
                num_lines=num_lines
            ))
        
        return regions
    
    def segment(self, image: np.ndarray) -> Tuple[List[Region], List[Region]]:
        """
        이미지에서 오선보와 TAB 영역 분리
        
        Returns:
            (staff_regions, tab_regions)
        """
        lines = self.detect_horizontal_lines(image)
        groups = self.group_lines(lines)
        regions = self.classify_groups(groups, image.shape[1])
        
        staff_regions = [r for r in regions if r.type == 'staff']
        tab_regions = [r for r in regions if r.type == 'tab']
        
        return staff_regions, tab_regions
    
    def extract_tab_only(self, image: np.ndarray) -> Optional[np.ndarray]:
        """TAB 영역만 추출"""
        staff_regions, tab_regions = self.segment(image)
        
        if not tab_regions:
            print("[Segmenter] No TAB region found, returning original")
            return image
        
        # 모든 TAB 영역 병합
        y_min = min(r.y_start for r in tab_regions)
        y_max = max(r.y_end for r in tab_regions)
        
        return image[y_min:y_max, :]
    
    def visualize(self, image: np.ndarray, 
                  staff_regions: List[Region], 
                  tab_regions: List[Region]) -> np.ndarray:
        """분리 결과 시각화"""
        vis = image.copy()
        
        # 오선보 영역: 파란색
        for region in staff_regions:
            cv2.rectangle(vis, 
                         (region.x_start, region.y_start), 
                         (region.x_end, region.y_end), 
                         (255, 0, 0), 2)
            cv2.putText(vis, f"STAFF ({region.num_lines} lines)", 
                       (region.x_start + 10, region.y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # TAB 영역: 녹색
        for region in tab_regions:
            cv2.rectangle(vis, 
                         (region.x_start, region.y_start), 
                         (region.x_end, region.y_end), 
                         (0, 255, 0), 2)
            cv2.putText(vis, f"TAB ({region.num_lines} lines)", 
                       (region.x_start + 10, region.y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis


def test_segmenter(image_path: str, output_path: str = None):
    """테스트 실행"""
    print(f"[Test] Loading {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"[Error] Cannot load image")
        return
    
    segmenter = RegionSegmenter()
    staff_regions, tab_regions = segmenter.segment(image)
    
    print(f"[Result] Found {len(staff_regions)} staff regions, {len(tab_regions)} TAB regions")
    
    for i, r in enumerate(staff_regions):
        print(f"  Staff {i+1}: y={r.y_start}-{r.y_end}, lines={r.num_lines}")
    
    for i, r in enumerate(tab_regions):
        print(f"  TAB {i+1}: y={r.y_start}-{r.y_end}, lines={r.num_lines}")
    
    # 시각화
    vis = segmenter.visualize(image, staff_regions, tab_regions)
    
    output_path = output_path or "segmentation_result.png"
    cv2.imwrite(output_path, vis)
    print(f"[Saved] {output_path}")
    
    # TAB만 추출
    tab_only = segmenter.extract_tab_only(image)
    if tab_only is not None:
        tab_path = output_path.replace(".png", "_tab_only.png")
        cv2.imwrite(tab_path, tab_only)
        print(f"[Saved] {tab_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_segmenter(sys.argv[1])
    else:
        # 기본 테스트
        test_segmenter("test_samples/images/page_1.png", "segmentation_result.png")
