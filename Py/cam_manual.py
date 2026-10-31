from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch

# ====================================
# MODELO
# ====================================

model = YOLO(
    r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"

)

net = model.model

# ====================================
# IMAGEN
# ====================================

img_path = r"..\dataset_trans\images\val\58b9d7b1-23d2-11e8-a6a3-ec086b02610b.jpg"

img = Image.open(img_path).convert("RGB")
img = img.resize((640,640))

img_np = np.array(img)

rgb_img = img_np.astype(np.float32) / 255.0

x = (
    torch.tensor(rgb_img)
    .permute(2,0,1)
    .unsqueeze(0)
)

x = x.float()

# GPU si existe
device = "cuda" if torch.cuda.is_available() else "cpu"

net = net.to(device)
x = x.to(device)

# ====================================
# FORWARD
# ====================================

with torch.no_grad():
    outputs = net(x)

feats = outputs[1]["feats"]

# usamos la feature profunda
feat = feats[2]

print(feat.shape)

# ====================================
# ACTIVATION MAP
# ====================================

cam = feat.mean(dim=1).squeeze()

cam = cam.cpu().numpy()

cam = np.maximum(cam,0)

cam = (cam - cam.min()) / (cam.max()-cam.min()+1e-8)

cam = cv2.resize(
    cam,
    (640,640)
)

heatmap = cv2.applyColorMap(
    np.uint8(255*cam),
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    img_np,
    0.6,
    heatmap,
    0.4,
    0
)

cv2.imwrite(
    "bobcat_cam.png",
    overlay
)

print("CAM GENERADO")