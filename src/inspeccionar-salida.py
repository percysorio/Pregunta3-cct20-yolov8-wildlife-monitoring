from ultralytics import YOLO
import torch

model = YOLO(
 r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"

)

net = model.model

x = torch.randn(1,3,640,640)

out = net(x)

print(type(out))
print(len(out))

print("\n===== ELEMENTO 0 =====")
print(type(out[0]))
print(out[0].shape)

print("\n===== ELEMENTO 1 =====")
print(type(out[1]))

if isinstance(out[1], dict):
    print(out[1].keys())