import json
from pathlib import Path
from collections import Counter

ANN = Path(r"..\eccv_18_annotations")

with open(ANN / "trans_val_annotations.json", "r") as f:
    data = json.load(f)

# índice de imágenes
img_lookup = {
    img["id"]: img
    for img in data["images"]
}

counter = Counter()

for ann in data["annotations"]:

    if "bbox" not in ann:
        continue

    img = img_lookup[ann["image_id"]]

    img_area = img["width"] * img["height"]

    x, y, w, h = ann["bbox"]

    bbox_area = w * h

    ratio = bbox_area / img_area

    if ratio < 0.01:
        counter["Pequeño"] += 1

    elif ratio < 0.05:
      counter["Mediano"] += 1

    else:
        counter["Grande"] += 1

total = sum(counter.values())

print("="*40)

for k, v in counter.items():

    print(
        f"{k}: {v} "
        f"({100*v/total:.2f}%)"
    )

print("="*40)