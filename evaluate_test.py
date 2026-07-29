import glob
import os

from ultralytics import YOLO

# train.py pins name="train", exist_ok=True, so training always overwrites
# runs/detect/train/weights/best.pt rather than creating train2, train3, ...
# Glob + pick-newest is kept as a safety net in case that ever changes.
candidates = glob.glob(os.path.join('.', 'runs', 'detect', 'train*', 'weights', 'best.pt'))

if not candidates:
    raise FileNotFoundError(
        "No trained model found under runs/detect/train*/weights/best.pt. Run train.py to train one first."
    )

model_path = max(candidates, key=os.path.getmtime)

model = YOLO(model_path)

metrics = model.val(data="config.yaml", split="test", name="test", exist_ok=True)

print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP75:    {metrics.box.map75:.4f}")
