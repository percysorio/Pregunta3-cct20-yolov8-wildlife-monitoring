from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================
# Cargar modelo
# ==========================

model = YOLO(
    r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"
)

net = model.model

# ==========================
# Imagen
# ==========================

img_path = r"..\dataset_trans\images\val\58b9d7b1-23d2-11e8-a6a3-ec086b02610b.jpg"
#\58b9d7b1-23d2-11e8-a6a3-ec086b02610b.jpg"
#5a14a3f4-23d2-11e8-a6a3-ec086b02610b.jpg"

img = Image.open(img_path).convert("RGB")

# Redimensionar
img = img.resize((640, 640))

img_np = np.array(img)

rgb_img = img_np.astype(np.float32) / 255.0

input_tensor = (
    torch.tensor(rgb_img)
    .permute(2,0,1)
    .unsqueeze(0)
)

# ==========================
# Capa objetivo
# ==========================

target_layers = [net.model[9]]

# ==========================
# Grad-CAM
# ==========================

cam = GradCAM(
    model=net,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=None
)

grayscale_cam = grayscale_cam[0]

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

# ==========================
# Guardar
# ==========================

cv2.imwrite(
    "bobcat_gradcam.jpg",
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print("Grad-CAM generado.")