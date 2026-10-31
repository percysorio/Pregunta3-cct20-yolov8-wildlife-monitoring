Wildlife Monitoring using YOLOv8s and Caltech Camera Traps (CCT20)
Author:Percy Marcovich Osorio Medina
Master's Degree in Artificial Intelligence

Universidad Nacional de Ingeniería (UNI)

-----------------------------------------------------------------------------------------------------------------------------------

Project Overview
This project implements an automated wildlife monitoring system based on YOLOv8s using the Caltech Camera Traps (CCT20) dataset.

The objective is to evaluate the ability of modern object detectors to identify wildlife species in camera trap images and to analyze their capacity to generalize to unseen camera locations.

The study includes:

Dataset preparation and COCO-to-YOLO conversion
YOLOv8s training and validation
Evaluation on CIS and TRANS benchmark partitions
Geographic generalization analysis
Object-size robustness analysis
Ecological activity reporting
Explainability through activation maps
------------------------------------------------------------------------------------------------------------------------------------

Dataset
Caltech Camera Traps (CCT20)

Dataset characteristics:
Feature Value

Total Images 243,100

Camera Locations 140

Categories 21 Species + Empty

Bounding Boxes ~66,000

Empty Images ~70%

Partitions used in this project:
Partition Images

Training (ECCV Train) 13,553

CIS Validation 3,484

TRANS Validation 1,725

------------------------------------------------------------------------------------------------------------------------------------

Model Configuration
Parameter Value
Architecture YOLOv8s

Resolution 640 × 640

Batch Size 8

Epochs 30

Framework PyTorch

Device CUDA

GPU NVIDIA RTX 5050 Laptop GPU

------------------------------------------------------------------------------------------------------------------------------------

Main Results
CIS Evaluation
Metric Value

Precision 0.815

Recall 0.781

mAP@0.5 0.847

mAP@0.5:0.95 0.614

TRANS Evaluation
Metric Value

Precision 0.527

Recall 0.544

mAP@0.5 0.506

mAP@0.5:0.95 0.366

------------------------------------------------------------------------------------------------------------------------------------

Key Finding
The detector achieved strong performance on the CIS partition but experienced a significant performance drop on the TRANS partition.

mAP@0.5 decreased from: 0.847 → 0.506

-------------------------------------------------------------------------------------------------------------------------------------

This result demonstrates the existence of a geographic generalization gap when the model is exposed to previously unseen camera locations.

-------------------------------------------------------------------------------------------------------------------------------------

Explainability
Activation maps were generated from deep YOLOv8s feature representations to visualize the detector's areas of attention.
The analysis showed that the model focuses primarily on the animal region while still exhibiting secondary activations in surrounding contextual areas.
--------------------------------------------------------------------------------------------------------------------------------------

Repository Structure
prepare_cct20.py
prepare_trans.py
train_yolo.py
evaluate_yolo.py
evaluate_trans.py
analyze_bbox_size.py
ecological_report.py
cam_manual.py
README.md
-------------------------------------------------------------------------------------------------------------------------------------

Installation
Create a Python environment and install dependencies:
-------------------------------------------------------------------------------------------------------------------------------------

pip install ultralytics

pip install torch torchvision

pip install numpy pandas matplotlib

pip install opencv-python pillow

-------------------------------------------------------------------------------------------------------------------------------------

Training
-------------------------------------------------------------------------------------------------------------------------------------

python train_yolo.py

-------------------------------------------------------------------------------------------------------------------------------------

TRANS Evaluation
-------------------------------------------------------------------------------------------------------------------------------------

python evaluate_trans.py

-------------------------------------------------------------------------------------------------------------------------------------

Ecological Monitoring Report
-------------------------------------------------------------------------------------------------------------------------------------

python ecological_report.py

-------------------------------------------------------------------------------------------------------------------------------------

Activation Maps
-------------------------------------------------------------------------------------------------------------------------------------

python cam_manual.py

-------------------------------------------------------------------------------------------------------------------------------------

Research Contributions
-------------------------------------------------------------------------------------------------------------------------------------

Complete YOLOv8s wildlife detection pipeline
Evaluation using the CCT20 benchmark
CIS vs TRANS geographic generalization analysis
Object-size robustness study
Ecological activity reporting module
Model interpretability through activation maps
