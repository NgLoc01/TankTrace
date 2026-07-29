from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model 
    model = YOLO("yolov8s.pt")

    # Use the model
    results = model.train(data="config.yaml", epochs=100, patience=20, degrees=15, name="train", exist_ok=True, device="mps")  # train the model
    #stop early if val stalls; degrees=15 adds mild rotation robustness; always overwrite runs/detect/train, device=mps uses the Mac GPU instead of CPU

#Took about 28 minutes to train the model with 100 epochs, 160 train, 20 val, 20 test, see results in runs/detect/train/results.csv second column(time) last cell 1693.6