"""
YOLO Training Script for TAB Recognition

Prerequisites:
    pip install ultralytics

Usage:
    1. Generate training data:
       python -m omnitab.training.synthetic_tab_generator --output training_data --samples 10000
    
    2. Train YOLO:
       python -m omnitab.training.train_yolo --data training_data/data.yaml --epochs 100
"""

import argparse
import sys
import os
from pathlib import Path

# Windows Unicode 출력 문제 해결 - tqdm 패치
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # tqdm을 ASCII 모드로 강제 패치
    try:
        from tqdm import tqdm as _tqdm
        from functools import partial
        import tqdm as tqdm_module
        
        # ASCII 전용 tqdm 래퍼
        class ASCIITqdm(_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs['ascii'] = True
                kwargs['ncols'] = 100
                super().__init__(*args, **kwargs)
        
        # 전역 패치
        tqdm_module.tqdm = ASCIITqdm
        tqdm_module.std.tqdm = ASCIITqdm
    except ImportError:
        pass
    
    # Windows 콘솔 UTF-8 설정
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except:
        pass


def train_yolo(data_yaml: str, epochs: int = 100, batch_size: int = 16, 
               model: str = "yolov8n.pt", imgsz: int = 640):
    """
    Train YOLO model on synthetic TAB data.
    
    Args:
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size
        model: Base model (yolov8n, yolov8s, yolov8m, etc.)
        imgsz: Image size for training
    """
    try:
        from ultralytics import YOLO
        # Windows에서 진행바 깨짐 방지 - ultralytics 내부 tqdm 패치
        if sys.platform == 'win32':
            try:
                import ultralytics.utils as ul_utils
                from tqdm import tqdm
                # ASCII 모드 tqdm으로 교체
                original_tqdm = tqdm
                def ascii_tqdm(*args, **kwargs):
                    kwargs['ascii'] = ' >=]'
                    kwargs['ncols'] = 120
                    return original_tqdm(*args, **kwargs)
                ul_utils.TQDM = ascii_tqdm
            except Exception as e:
                print(f"[Warning] tqdm patch failed: {e}")
    except ImportError:
        print("Error: ultralytics not installed!")
        print("Run: pip install ultralytics")
        return
    
    # Load base model
    print(f"[Train] Loading base model: {model}")
    yolo = YOLO(model)
    
    # Train
    print(f"[Train] Starting training...")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project="runs/tab_detection",
        name="yolo_tab",
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=False,  # 진행바 비활성화 (Windows 터미널 호환성)
    )
    
    print(f"\n[Train] Complete!")
    print(f"  Best model: {results.best}")
    
    return results


def validate_yolo(model_path: str, data_yaml: str):
    """Validate trained model"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed!")
        return
    
    yolo = YOLO(model_path)
    results = yolo.val(data=data_yaml)
    
    print(f"\n[Validation Results]")
    print(f"  mAP50: {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    
    return results


def predict_tab(model_path: str, image_path: str, conf: float = 0.5):
    """
    Predict TAB notes in an image.
    
    Args:
        model_path: Path to trained YOLO model
        image_path: Path to TAB image
        conf: Confidence threshold
        
    Returns:
        List of detected notes with positions
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed!")
        return []
    
    yolo = YOLO(model_path)
    results = yolo(image_path, conf=conf)
    
    # Parse results
    notes = []
    class_names = [str(i) for i in range(25)] + ['h', 'p', 'x', 'harmonic']
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calculate center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            notes.append({
                'class_id': cls,
                'class_name': class_names[cls] if cls < len(class_names) else str(cls),
                'confidence': conf,
                'x': x_center,
                'y': y_center,
                'bbox': (x1, y1, x2, y2)
            })
    
    # Sort by x position (left to right)
    notes.sort(key=lambda n: n['x'])
    
    return notes


def notes_to_tab(notes: list, img_height: int, num_strings: int = 6) -> list:
    """
    Convert detected notes to TAB format.
    
    Uses Y position to determine which string (1-6).
    
    Args:
        notes: List of detected notes
        img_height: Image height for calculating string positions
        num_strings: Number of strings (default 6)
        
    Returns:
        List of (string, fret, x_position) tuples
    """
    # Estimate line positions (assume even spacing)
    margin_ratio = 0.15
    usable_height = img_height * (1 - 2 * margin_ratio)
    line_spacing = usable_height / (num_strings - 1)
    first_line_y = img_height * margin_ratio
    
    tab_notes = []
    
    for note in notes:
        y = note['y']
        
        # Find nearest string
        string = 1
        min_dist = abs(y - first_line_y)
        
        for s in range(2, num_strings + 1):
            line_y = first_line_y + (s - 1) * line_spacing
            dist = abs(y - line_y)
            if dist < min_dist:
                min_dist = dist
                string = s
        
        # Get fret (class_id for 0-24, -1 for special)
        if note['class_id'] <= 24:
            fret = note['class_id']
        elif note['class_name'] == 'x':
            fret = -1
        else:
            fret = note.get('fret', 0)
        
        tab_notes.append({
            'string': string,
            'fret': fret,
            'x': note['x'],
            'confidence': note['confidence'],
            'class_name': note['class_name']
        })
    
    return tab_notes


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for TAB recognition")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data", "-d", required=True, help="Path to data.yaml")
    train_parser.add_argument("--epochs", "-e", type=int, default=100)
    train_parser.add_argument("--batch", "-b", type=int, default=16)
    train_parser.add_argument("--model", "-m", default="yolov8n.pt")
    train_parser.add_argument("--imgsz", type=int, default=640)
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model")
    val_parser.add_argument("--model", "-m", required=True, help="Model path")
    val_parser.add_argument("--data", "-d", required=True, help="Path to data.yaml")
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict on image")
    pred_parser.add_argument("--model", "-m", required=True, help="Model path")
    pred_parser.add_argument("--image", "-i", required=True, help="Image path")
    pred_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_yolo(args.data, args.epochs, args.batch, args.model, args.imgsz)
    elif args.command == "validate":
        validate_yolo(args.model, args.data)
    elif args.command == "predict":
        notes = predict_tab(args.model, args.image, args.conf)
        print(f"\nDetected {len(notes)} notes:")
        for n in notes:
            print(f"  {n['class_name']} at ({n['x']:.0f}, {n['y']:.0f}) conf={n['confidence']:.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
