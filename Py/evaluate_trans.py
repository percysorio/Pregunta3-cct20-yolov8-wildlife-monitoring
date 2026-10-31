from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == "__main__":

    freeze_support()

    model = YOLO(
        r"runs\detect\cct20_yolov8s\weights\best.pt"
    )

    metrics = model.val(
        data="dataset_trans/trans.yaml",
        device=0,
        workers=0
    )

    print("=" * 50)
    print("mAP50      :", metrics.box.map50)
    print("mAP50-95   :", metrics.box.map)
    print("Precision  :", metrics.box.mp)
    print("Recall     :", metrics.box.mr)
    print("=" * 50)