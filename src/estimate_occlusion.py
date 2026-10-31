import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =====================================================
# CONFIGURACIÓN
# =====================================================

MODEL_PATH = r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"

IMAGE_DIR = r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\dataset_yolo\images\val"

# =====================================================
# CARGAR MODELO
# =====================================================

print("Cargando modelo...")

model = YOLO(MODEL_PATH)

# =====================================================
# FUNCIÓN DE OCLUSIÓN
# =====================================================

def estimate_occlusion(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / edges.size

    occlusion_score = 1 - (edge_density * 4)

    occlusion_score = np.clip(occlusion_score, 0, 1)

    return float(occlusion_score)

# =====================================================
# LEER IMÁGENES
# =====================================================

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Imágenes encontradas: {len(images)}")

# prueba rápida
#images = images[:100]

results_data = []

# =====================================================
# PROCESAMIENTO
# =====================================================

for idx, image_name in enumerate(images):

    print(f"[{idx+1}/{len(images)}] {image_name}")

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    h_img, w_img = image.shape[:2]

    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        if len(result.boxes) == 0:
            continue

        for box in result.boxes:

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            bbox_area = (x2 - x1) * (y2 - y1)

            image_area = w_img * h_img

            relative_area = bbox_area / image_area

            occlusion_score = estimate_occlusion(crop)

            if occlusion_score < 0.33:
                occlusion_level = "Baja"
            elif occlusion_score < 0.66:
                occlusion_level = "Media"
            else:
                occlusion_level = "Alta"

            results_data.append({
                "image": image_name,
                "class": cls,
                "confidence": conf,
                "relative_area": relative_area,
                "occlusion_score": occlusion_score,
                "occlusion_level": occlusion_level
            })

# =====================================================
# RESULTADOS
# =====================================================

print()
print("Detecciones encontradas:", len(results_data))

if len(results_data) == 0:
    print("No se encontraron detecciones.")
    quit()

df = pd.DataFrame(results_data)

# =====================================================
# GUARDAR ARCHIVOS
# =====================================================

df.to_csv("occlusion_analysis.csv", index=False)

summary = (
    df.groupby("occlusion_level")
    .size()
    .reset_index(name="detections")
)

summary.to_csv("occlusion_summary.csv", index=False)

print()
print(summary)

print()
print("Archivo generado: occlusion_analysis.csv")
print("Archivo generado: occlusion_summary.csv")