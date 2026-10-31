# evaluate_trans_simple.py

from ultralytics import YOLO

if __name__ == "__main__":

    print("Cargando modelo...")

    model = YOLO(
        r"runs\detect\runs\cct20_yolov8s-3\weights\best.pt"
    )

    print("Modelo cargado")

    metrics = model.val(
        data="dataset_trans/trans.yaml",
        device=0,
        workers=0,
        batch=8
    )

    print("\nRESULTADOS TRANS")
    print("="*40)
    print("mAP50      :", metrics.box.map50)
    print("mAP50-95   :", metrics.box.map)
    print("Precision  :", metrics.box.mp)
    print("Recall     :", metrics.box.mr)
    print("="*40)