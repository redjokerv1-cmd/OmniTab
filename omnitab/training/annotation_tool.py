"""
TAB Image Annotation Tool

실제 TAB 이미지를 클릭해서 annotation하는 도구
YOLO 형식으로 저장하여 learning-data-vault에 추가

Usage:
    python -m omnitab.training.annotation_tool --image path/to/tab.png
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class TabAnnotationTool:
    """
    Interactive TAB annotation tool using OpenCV
    
    클릭으로 숫자 위치 지정, 키보드로 프렛 번호 입력
    """
    
    # 클래스 정의 (YOLO 형식)
    CLASSES = [str(i) for i in range(25)] + ['h', 'p', 'x', 'harmonic']
    
    def __init__(self, image_path: str, output_dir: str = None, load_existing: bool = True):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir) if output_dir else self.image_path.parent
        
        # 이미지 로드
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.height, self.width = self.image.shape[:2]
        self.display_image = self.image.copy()
        
        # Annotations
        self.annotations: List[Dict] = []
        self.current_class = "0"  # 현재 선택된 클래스
        
        # 기존 annotation 로드
        if load_existing:
            self._load_existing_annotations()
        
        # UI state
        self.window_name = "TAB Annotation Tool"
        self.mouse_pos = (0, 0)
        self.drawing = False
        self.start_pos = None
        
        # Box size (default)
        self.box_width = 20
        self.box_height = 20
    
    def _load_existing_annotations(self):
        """기존 annotation JSON 파일이 있으면 로드"""
        json_path = self.output_dir / f"{self.image_path.stem}_annotation.json"
        txt_path = self.output_dir / f"{self.image_path.stem}.txt"
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', [])
                    print(f"[Loaded] {len(self.annotations)} annotations from {json_path}")
            except Exception as e:
                print(f"[Warning] Failed to load {json_path}: {e}")
        elif txt_path.exists():
            # YOLO txt 파일에서 로드
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            class_name = self.CLASSES[class_id] if class_id < len(self.CLASSES) else str(class_id)
                            
                            self.annotations.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': w,
                                'height': h,
                                'pixel_x': int(x_center * self.width),
                                'pixel_y': int(y_center * self.height)
                            })
                print(f"[Loaded] {len(self.annotations)} annotations from {txt_path}")
            except Exception as e:
                print(f"[Warning] Failed to load {txt_path}: {e}")
        
    def run(self):
        """메인 루프"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(1400, self.width), min(900, self.height))
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("TAB Annotation Tool")
        print("=" * 60)
        print("\n[Controls]")
        print("  Left Click: Add annotation at position")
        print("  Right Click: Remove nearest annotation")
        print("  Middle Click: Edit nearest annotation's class")
        print("  0-9: Set fret number (0-9)")
        print("  Shift+0-9: Set fret number (10-19)")
        print("  Ctrl+0-4: Set fret number (20-24)")
        print("  h: Set class to 'harmonic'")
        print("  p: Set class to 'pull-off'")
        print("  x: Set class to 'mute'")
        print("  e: Edit nearest annotation (change to current class)")
        print("  +/-: Adjust box size")
        print("  s: Save annotations")
        print("  u: Undo last annotation")
        print("  q: Quit")
        print("\n" + "=" * 60)
        
        while True:
            self._draw()
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save()
            elif key == ord('u'):
                self._undo()
            elif key == ord('h'):
                self.current_class = "harmonic"
                print(f"[Class] harmonic")
            elif key == ord('p'):
                self.current_class = "p"
                print(f"[Class] pull-off")
            elif key == ord('x'):
                self.current_class = "x"
                print(f"[Class] mute")
            elif key == ord('e'):
                # 가장 가까운 annotation 수정
                self._edit_nearest(self.mouse_pos[0], self.mouse_pos[1])
            elif key == ord('+') or key == ord('='):
                self.box_width += 2
                self.box_height += 2
                print(f"[Box] {self.box_width}x{self.box_height}")
            elif key == ord('-'):
                self.box_width = max(10, self.box_width - 2)
                self.box_height = max(10, self.box_height - 2)
                print(f"[Box] {self.box_width}x{self.box_height}")
            elif ord('0') <= key <= ord('9'):
                self.current_class = chr(key)
                print(f"[Class] {self.current_class}")
        
        cv2.destroyAllWindows()
        
        # 자동 저장
        if self.annotations:
            save = input("\nSave annotations before exit? (y/n): ")
            if save.lower() == 'y':
                self._save()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리"""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 왼쪽 클릭: annotation 추가
            self._add_annotation(x, y)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 오른쪽 클릭: 가장 가까운 annotation 삭제
            self._remove_nearest(x, y)
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            # 중간 클릭: 가장 가까운 annotation 수정
            self._edit_nearest(x, y)
    
    def _add_annotation(self, x: int, y: int):
        """Annotation 추가"""
        # YOLO 형식: class x_center y_center width height (normalized)
        x_center = x / self.width
        y_center = y / self.height
        w = self.box_width / self.width
        h = self.box_height / self.height
        
        class_id = self.CLASSES.index(self.current_class) if self.current_class in self.CLASSES else 0
        
        annotation = {
            'class_id': class_id,
            'class_name': self.current_class,
            'x_center': x_center,
            'y_center': y_center,
            'width': w,
            'height': h,
            'pixel_x': x,
            'pixel_y': y
        }
        
        self.annotations.append(annotation)
        print(f"[Added] {self.current_class} at ({x}, {y}) - Total: {len(self.annotations)}")
    
    def _remove_nearest(self, x: int, y: int):
        """가장 가까운 annotation 삭제"""
        if not self.annotations:
            return
        
        # 가장 가까운 찾기
        min_dist = float('inf')
        min_idx = -1
        
        for i, ann in enumerate(self.annotations):
            px, py = ann['pixel_x'], ann['pixel_y']
            dist = (px - x) ** 2 + (py - y) ** 2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        if min_idx >= 0 and min_dist < 50 ** 2:  # 50픽셀 이내
            removed = self.annotations.pop(min_idx)
            print(f"[Removed] {removed['class_name']} - Total: {len(self.annotations)}")
    
    def _undo(self):
        """마지막 annotation 취소"""
        if self.annotations:
            removed = self.annotations.pop()
            print(f"[Undo] {removed['class_name']} - Total: {len(self.annotations)}")
    
    def _edit_nearest(self, x: int, y: int):
        """가장 가까운 annotation의 클래스 수정"""
        if not self.annotations:
            print("[Warning] No annotations to edit")
            return
        
        # 가장 가까운 찾기
        min_dist = float('inf')
        min_idx = -1
        
        for i, ann in enumerate(self.annotations):
            px, py = ann['pixel_x'], ann['pixel_y']
            dist = (px - x) ** 2 + (py - y) ** 2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        if min_idx >= 0 and min_dist < 100 ** 2:  # 100픽셀 이내
            old_class = self.annotations[min_idx]['class_name']
            class_id = self.CLASSES.index(self.current_class) if self.current_class in self.CLASSES else 0
            
            self.annotations[min_idx]['class_id'] = class_id
            self.annotations[min_idx]['class_name'] = self.current_class
            print(f"[Edited] {old_class} -> {self.current_class} at ({self.annotations[min_idx]['pixel_x']}, {self.annotations[min_idx]['pixel_y']})")
        else:
            print("[Warning] No annotation nearby to edit")
    
    def _draw(self):
        """화면 그리기"""
        self.display_image = self.image.copy()
        
        # Annotations 그리기
        for ann in self.annotations:
            x = int(ann['x_center'] * self.width)
            y = int(ann['y_center'] * self.height)
            w = int(ann['width'] * self.width)
            h = int(ann['height'] * self.height)
            
            # 박스
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨
            label = ann['class_name']
            cv2.putText(self.display_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 현재 마우스 위치에 미리보기
        mx, my = self.mouse_pos
        x1 = mx - self.box_width // 2
        y1 = my - self.box_height // 2
        x2 = mx + self.box_width // 2
        y2 = my + self.box_height // 2
        cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # 상태 표시
        status = f"Class: {self.current_class} | Annotations: {len(self.annotations)} | Box: {self.box_width}x{self.box_height}"
        cv2.putText(self.display_image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _save(self):
        """YOLO 형식으로 저장"""
        if not self.annotations:
            print("[Warning] No annotations to save")
            return
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.image_path.stem
        
        # YOLO txt 파일
        txt_path = self.output_dir / f"{base_name}.txt"
        with open(txt_path, 'w') as f:
            for ann in self.annotations:
                line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                f.write(line)
        
        print(f"[Saved] {txt_path}")
        
        # JSON 백업 (상세 정보)
        json_path = self.output_dir / f"{base_name}_annotation.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image': str(self.image_path),
                'width': self.width,
                'height': self.height,
                'annotations': self.annotations,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"[Saved] {json_path}")
        print(f"[Total] {len(self.annotations)} annotations")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TAB Image Annotation Tool")
    parser.add_argument("--image", "-i", required=True, help="Path to TAB image")
    parser.add_argument("--output", "-o", help="Output directory (default: same as image)")
    
    args = parser.parse_args()
    
    tool = TabAnnotationTool(args.image, args.output)
    tool.run()


if __name__ == "__main__":
    main()
