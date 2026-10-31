from ultralytics import YOLO

model = YOLO(
   r"C:\Users\PERCY OSORIO\Documents\Maestria Inteligencia Artificial\Ciclo 2\2.-Redes Neuronales y Aprendizaje Profundo\Examen Parcial\Pregunta3\runs\detect\runs\cct20_yolov8s-3\weights\best.pt"

)

img_path = r"..\dataset_trans\images\val\58b9d7b1-23d2-11e8-a6a3-ec086b02610b.jpg"


results = model(img_path)

print(results[0].boxes.cls)
print(results[0].boxes.conf)