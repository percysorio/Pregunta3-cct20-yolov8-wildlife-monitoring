import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path.cwd()

IMAGES_DIR = ROOT.parent / "eccv_18_all_images_sm"
ANN_DIR = ROOT.parent / "eccv_18_annotations"

TRANS_JSON = ANN_DIR / "trans_val_annotations.json"

TRANS_DIR = ROOT / "dataset_trans"

(TRANS_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
(TRANS_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)


def coco_to_yolo(bbox, width, height):
    x, y, w, h = bbox

    return (
        (x + w / 2) / width,
        (y + h / 2) / height,
        w / width,
        h / height,
    )


with open(TRANS_JSON) as f:
    data = json.load(f)

cat_map = {}

for idx, cat in enumerate(data["categories"]):
    cat_map[cat["id"]] = idx

image_lookup = {
    img["id"]: img for img in data["images"]
}

anns_per_image = defaultdict(list)

for ann in data["annotations"]:
    anns_per_image[ann["image_id"]].append(ann)

for img in data["images"]:

    src = IMAGES_DIR / img["file_name"]
    dst = TRANS_DIR / "images" / "val" / img["file_name"]

    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

for image_id, anns in anns_per_image.items():

    img = image_lookup[image_id]

    txt_file = (
        TRANS_DIR / "labels" / "val"
        / (Path(img["file_name"]).stem + ".txt")
    )

    lines = []

    for ann in anns:

        if "bbox" not in ann:
            continue

        x, y, w, h = coco_to_yolo(
            ann["bbox"],
            img["width"],
            img["height"]
        )

        cls = cat_map[ann["category_id"]]

        lines.append(
            f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    with open(txt_file, "w") as f:
        f.write("\n".join(lines))

with open(TRANS_DIR / "trans.yaml", "w") as f:

    f.write(f"path: {TRANS_DIR.resolve().as_posix()}\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")

    for idx, c in enumerate(data["categories"]):
        f.write(f"  {idx}: {c['name']}\n")

print("TRANS listo")