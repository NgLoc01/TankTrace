# TankTrace

**Won't run** `data/`, `videos/`, `runs/`, and `yolov8s.pt`
are gitignored (too large to track) and missing from this repo. Add your
own dataset and video, then run `train.py` to generate `runs/`.

## Summary
Fine-tunes YOLOv8s to detect a single custom class, `sherman_tank`, then runs
that model on video files or a live screen capture.

Screenshots were taken from War Thunder and labeled in CVAT, then split into
train/validation/test sets. A base YOLOv8 model was downloaded and `train.py`
fine-tuned it on this dataset, producing `best.pt` which the trained weights used to actually detect tanks and judge how well the model learned. `predict_trained.py` runs the trained model on a video to see it in action, while `predict_untrained.py` runs the same video through the stock (untrained) YOLOv8 model for comparison. `evaluate_test.py` scores the trained model against the test set to measure accuracy.


## Inspiration:
- https://www.youtube.com/watch?v=m9fH9OWn8YM&list=LL&index=5&t=1959s
- https://github.com/ultralytics/ultralytics

Download and tools:
- https://github.com/ultralytics/assets/releases/tag/v8.4.0
- https://www.cvat.ai



## YOLOv8
The stock YOLOv8 model is trained on the COCO dataset, which has 80 classes
like `boat`, `dog`, `car`, etc. There are no `sherman_tank` class. I fine-tuned the
model on my own labeled dataset to detect sherman tanks; the result is a
single-class model with just 1 class, `sherman_tank`, rather than the
original 80.


## Setup
`python3 -m venv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`

`predict_trained.py` and `predict_untrained.py` also shell out to `ffmpeg` to
re-mux audio onto the annotated video, so it must be installed and on your
`PATH` (`brew install ffmpeg` on macOS).


## Dataset layout
Not tracked in git (see `.gitignore`) — provide your own images/labels in
YOLO format under:

data/
  images/{train,val,test}/
  labels/{train,val,test}/

Classes and split paths are defined in `config.yaml` (one class:
`sherman_tank`).



## How to train
`python3 train.py`

Fine-tunes from the base `yolov8s.pt` weights (COCO-pretrained) on
`config.yaml` for up to 100 epochs, stopping early after 20 epochs without
validation improvement. Uses the Mac GPU (`device="mps"`). Weights are
written to `runs/detect/train/weights/best.pt` (and `last.pt`), always
overwriting the same `train` run folder rather than creating `train2`,
`train3`, etc.

`yolov8s.pt` isn't tracked in git (see `.gitignore`) — the first run of
`train.py` or `predict_untrained.py` downloads it automatically via the
`ultralytics` package. To fetch it manually instead:
https://github.com/ultralytics/assets/releases/tag/v8.4.0



## How to evaluate
`python evaluate_test.py`

Runs the most recently trained model against the held-out `test` split and
prints mAP50-95, mAP50, and mAP75.



## How to run detection on video
Both scripts expect a video at `videos/tank1.mp4` and write an annotated
copy alongside it.

`python predict_trained.py`     # uses runs/detect/train*/weights/best.pt
`python predict_untrained.py`   # uses the base yolov8s.pt, for before/after comparison

Output: `videos/tank1.mp4_trained_out.mp4` / `videos/tank1.mp4_untrained_out.mp4`.



## How to run live screen detection
`python3 screen_detect.py`

Captures a region of your screen (`capture_region` in the script, currently
the top-left quarter of a 3024x1964 display — adjust to match your setup)
and runs the trained model on it in real time. Press `q` or close the
preview window to stop.



