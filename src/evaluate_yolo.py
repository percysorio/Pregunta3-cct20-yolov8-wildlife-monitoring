from ultralytics import YOLO

model = YOLO(
    "runs/cct20_yolov8s/weights/best.pt"
)

metrics = model.val(
    data="dataset_yolo/cct20.yaml"
)

print("================================")
print("mAP50      :", metrics.box.map50)
print("mAP50-95   :", metrics.box.map)
print("Precision  :", metrics.box.mp)
print("Recall     :", metrics.box.mr)
print("================================")