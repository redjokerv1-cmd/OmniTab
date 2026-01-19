"""
YOLO Training Runner - Windows 호환 버전
ultralytics 내부 TQDM을 ascii=True로 패치
"""
import sys
import os

# ultralytics import 전에 TQDM 패치
print("[Init] Patching tqdm for ASCII mode...")

# 1. 표준 tqdm 패치
import tqdm as tqdm_module
from tqdm import tqdm as OriginalTqdm

class ASCIITqdm(OriginalTqdm):
    """ASCII 전용 tqdm"""
    def __init__(self, *args, **kwargs):
        kwargs['ascii'] = True  # ASCII 문자만 사용
        kwargs.setdefault('ncols', 100)
        super().__init__(*args, **kwargs)

# 전역 패치
tqdm_module.tqdm = ASCIITqdm
tqdm_module.std.tqdm = ASCIITqdm
try:
    import tqdm.auto
    tqdm.auto.tqdm = ASCIITqdm
except:
    pass

# 2. ultralytics 내부 TQDM 패치
import ultralytics.utils as ul_utils

class UltralyticsASCIITqdm(OriginalTqdm):
    """ultralytics용 ASCII tqdm"""
    def __init__(self, *args, **kwargs):
        kwargs['ascii'] = True
        kwargs.setdefault('ncols', 100)
        kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        super().__init__(*args, **kwargs)

# ultralytics TQDM 교체
ul_utils.TQDM = UltralyticsASCIITqdm

print("[Init] tqdm patched successfully!")
print()

# 이제 YOLO import
from ultralytics import YOLO

def main():
    data_yaml = "training_data_full/data.yaml"
    epochs = 50
    batch_size = 16
    
    print("=" * 60)
    print("YOLO TAB Recognition Training")
    print("=" * 60)
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {batch_size}")
    print(f"  Device: CPU")
    print("=" * 60)
    print()
    
    yolo = YOLO("yolov8n.pt")
    
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        project="runs/tab_detection",
        name="yolo_tab_ascii",
        patience=20,
        save=True,
        plots=True,
    )
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print(f"  Best model: runs/tab_detection/yolo_tab_ascii/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
