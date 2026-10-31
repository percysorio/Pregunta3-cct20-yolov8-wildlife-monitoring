from ultralytics import YOLO

model = YOLO(
    r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"
)

for i, layer in enumerate(model.model.model):
    print(i, layer.__class__.__name__)