import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

if __name__ == "__main__":

    model = YOLO(
        r"runs\detect\runs\cct20_yolov8s-3\weights\best.pt"
    )

    image_dir = Path("dataset_trans/images/val")

    species_counter = Counter()

    total_images = 0
    images_with_animals = 0
    empty_images = 0

    for img in image_dir.glob("*.jpg"):

        total_images += 1

        result = model.predict(
            source=str(img),
            conf=0.25,
            verbose=False
        )[0]

        boxes = result.boxes

        if len(boxes) == 0:

            empty_images += 1

        else:

            images_with_animals += 1

            for cls in boxes.cls:

                species = result.names[int(cls)]

                species_counter[species] += 1

    report = pd.DataFrame(
        species_counter.items(),
        columns=["Species", "Detections"]
    )

    report = report.sort_values(
        by="Detections",
        ascending=False
    )

    report.to_csv(
        "ecological_report.csv",
        index=False
    )

    summary = pd.DataFrame({
        "Metric": [
            "Total Images",
            "Images With Animals",
            "Empty Images",
            "Empty Rate (%)",
            "Unique Species"
        ],
        "Value": [
            total_images,
            images_with_animals,
            empty_images,
            round(
                100 * empty_images / total_images,
                2
            ),
            len(species_counter)
        ]
    })

    summary.to_csv(
        "ecological_summary.csv",
        index=False
    )

    print("="*50)
    print(summary)
    print("="*50)

    print(report)