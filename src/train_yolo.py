from ultralytics import YOLO
import torch

if __name__ == "__main__":

    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset_yolo/cct20.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        cache=False,
        project="runs",
        name="cct20_yolov8s"
    )