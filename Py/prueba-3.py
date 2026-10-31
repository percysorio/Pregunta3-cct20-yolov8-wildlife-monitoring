from ultralytics import YOLO
import torch

model = YOLO(
    r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"

)

net = model.model

x = torch.randn(1,3,640,640)

out = net(x)

print(type(out))

if isinstance(out, tuple):
    print("TUPLE")
    print("Cantidad:", len(out))

    for i, item in enumerate(out):
        print(i, type(item))

        if hasattr(item, "shape"):
            print(item.shape)