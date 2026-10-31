from ultralytics import YOLO
import torch

model = YOLO(
    r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"
)

net = model.model

x = torch.randn(1,3,640,640)

out = net(x)

feats = out[1]["feats"]

print(type(feats))

if isinstance(feats, (list, tuple)):
    print("Cantidad:", len(feats))

    for i, f in enumerate(feats):
        print("\nFEATURE", i)

        if hasattr(f, "shape"):
            print(f.shape)