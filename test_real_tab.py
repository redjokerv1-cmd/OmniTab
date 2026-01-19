"""Test YOLO model on real PDF TAB"""
from ultralytics import YOLO

# 최종 훈련된 모델 로드
model = YOLO('runs/detect/runs/tab_detection/yolo_tab_ascii/weights/best.pt')

# 테스트 이미지 - 실제 PDF TAB
results = model('test_samples/images/page_1.png', conf=0.25, save=True)

print('='*60)
print('YOLO TAB Detection - Real PDF Test')
print('='*60)

for r in results:
    boxes = r.boxes
    print(f'Total detected: {len(boxes)} objects')
    print(f'Image size: {r.orig_shape}')
    
    # 클래스별 카운트
    class_counts = {}
    positions = []
    
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = r.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        positions.append({
            'class': cls_name,
            'conf': conf,
            'x': x_center,
            'y': y_center
        })
    
    print(f'\nClass distribution:')
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f'  {cls_name:>10}: {count}')
    
    # Y좌표로 줄(string) 그룹핑 시도
    print(f'\nSample detections (sorted by Y, top 20):')
    positions.sort(key=lambda p: p['y'])
    for p in positions[:20]:
        print(f"  {p['class']:>10} at Y={p['y']:.0f}, X={p['x']:.0f} (conf={p['conf']:.2f})")

print(f'\nResult image saved!')
