import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# =====================================================
# RUTAS
# =====================================================

ROOT = Path.cwd()

IMAGES_DIR = ROOT.parent / "eccv_18_all_images_sm"
ANN_DIR = ROOT.parent / "eccv_18_annotations"

TRAIN_JSON = ANN_DIR / "train_annotations.json"
VAL_JSON = ANN_DIR / "cis_val_annotations.json"

YOLO_DIR = ROOT / "dataset_yolo"

# =====================================================
# ESTRUCTURA YOLO
# =====================================================

for p in [
    YOLO_DIR / "images" / "train",
    YOLO_DIR / "images" / "val",
    YOLO_DIR / "labels" / "train",
    YOLO_DIR / "labels" / "val",
]:
    p.mkdir(parents=True, exist_ok=True)

# =====================================================
# FUNCION COCO -> YOLO
# =====================================================

def coco_to_yolo(bbox, width, height):

    x, y, w, h = bbox

    x_center = (x + w/2) / width
    y_center = (y + h/2) / height

    w = w / width
    h = h / height

    return x_center, y_center, w, h

# =====================================================
# PROCESAR SPLIT
# =====================================================

def process_split(json_file, split_name, save_categories=False):

    print(f"\nProcesando {split_name}...")

    with open(json_file, "r") as f:
        data = json.load(f)

    categories = data["categories"]

    cat_id_to_yolo = {}

    for idx, cat in enumerate(categories):
        cat_id_to_yolo[cat["id"]] = idx

    image_lookup = {}

    for img in data["images"]:
        image_lookup[img["id"]] = img

    ann_per_image = defaultdict(list)

    for ann in data["annotations"]:
        ann_per_image[ann["image_id"]].append(ann)

    img_dir = YOLO_DIR / "images" / split_name
    lbl_dir = YOLO_DIR / "labels" / split_name

    for img in tqdm(data["images"]):

        src = IMAGES_DIR / img["file_name"]
        dst = img_dir / img["file_name"]

        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    for image_id, anns in tqdm(ann_per_image.items()):

        img_info = image_lookup[image_id]

        width = img_info["width"]
        height = img_info["height"]

        txt_path = lbl_dir / (
            Path(img_info["file_name"]).stem + ".txt"
        )

        lines = []

        for ann in anns:

            if "bbox" not in ann:
                continue

            cls = cat_id_to_yolo[
                ann["category_id"]
            ]

            xc, yc, bw, bh = coco_to_yolo(
                ann["bbox"],
                width,
                height
            )

            lines.append(
                f"{cls} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}"
            )

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    if save_categories:
        return categories

# =====================================================
# TRAIN
# =====================================================

categories = process_split(
    TRAIN_JSON,
    "train",
    save_categories=True
)

# =====================================================
# VAL
# =====================================================

process_split(
    VAL_JSON,
    "val"
)

# =====================================================
# YAML
# =====================================================

yaml_path = YOLO_DIR / "cct20.yaml"

with open(yaml_path, "w") as f:

    f.write(f"path: {YOLO_DIR.resolve().as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")

    f.write("names:\n")

    for idx, cat in enumerate(categories):
        f.write(
            f"  {idx}: {cat['name']}\n"
        )

print("\nDataset YOLO listo ✅")
print(yaml_path)