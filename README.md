# Packaging Quality Inspection Classification Using YOLOv10

## Overview

This project implements a computer vision-based packaging quality inspection system using YOLOv10 to classify ginger candy packaging conditions.

The system was developed as part of an undergraduate thesis project to assist local MSME (UMKM) ginger candy producers in automating the packaging inspection process and improving quality control efficiency.

---

## Problem Statement

Manual packaging inspection is often:

* Time-consuming
* Inconsistent between operators
* Prone to human error
* Difficult to scale during high production demand

This project addresses these challenges by utilizing a YOLOv10 object detection model capable of automatically identifying packaging conditions from captured images.

---

## Objectives

* Automate packaging quality inspection
* Reduce human inspection workload
* Improve consistency of quality control
* Support production efficiency in MSME environments

---

## Classification Categories

The model classifies packaging into three categories:

| Class      | Description                                               |
| ---------- | --------------------------------------------------------- |
| Layak Edar | Packaging is in good condition and ready for distribution |
| Penyok     | Packaging is dented or deformed                           |
| Sobek      | Packaging is torn or physically damaged                   |

---

## Technology Stack

* Python
* YOLOv10
* Ultralytics
* OpenCV
* NumPy
* Google Colab

---

## Dataset

A custom dataset was collected and manually annotated from real ginger candy packaging products.

Dataset Characteristics:

* Real production packaging samples
* Three packaging condition classes
* Object detection annotations
* Custom-trained for packaging inspection use cases

---

## Model Training

The model was trained using YOLOv10 on a custom packaging dataset.

Training Pipeline:

1. Data Collection
2. Data Annotation
3. Dataset Preparation
4. YOLOv10 Training
5. Validation & Evaluation
6. Deployment Testing

---

## Results

### Performance Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | ~99%  |
| Recall    | ~99%  |
| mAP50     | ~99%  |
| mAP50-95  | ~99%  |

The model achieved high detection performance across all packaging categories and demonstrated reliable classification capability for real-world packaging inspection scenarios.

---

## Sample Prediction Results

### Layak Edar Detection

<img width="1920" height="1920" alt="val_batch0_labels" src="https://github.com/user-attachments/assets/0d034b12-4a80-47f2-8450-b97acd51511d" />


### Penyok Detection

<img width="1920" height="1920" alt="val_batch1_labels" src="https://github.com/user-attachments/assets/47c1ff5f-4c40-44b2-bd48-d65d20bc31ab" />


### Sobek Detection

<img width="1920" height="1920" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/82438785-75b4-4745-9515-c372155851a1" />


---

## Project Structure

```text
packaging-quality-inspection-classification-yolo/
│
├── app_skripsi.py
├── final_app.py
├── training.py
├── test.py
├── kamera.py
├── data.yaml
├── requirements.txt
├── README.md
│
├── train/
├── valid/
├── test/
│
└── assets/
```

## Installation

```bash
git clone https://github.com/akhsannurramdhan/packaging-quality-inspection-classification-yolo.git

cd packaging-quality-inspection-classification-yolo

pip install -r requirements.txt
```

## Run Detection

```bash
python app_skripsi.py
```

## Thesis Information

Title:

Implementation of YOLO Algorithm for Classification of Marketable Ginger Candy Packaging

Research Domain:

Computer Vision • Machine Learning • Quality Inspection

---

## Future Improvements

* Real-time camera deployment
* Mobile integration
* Larger dataset collection
* Multi-product packaging inspection
* Production line integration

---

## Author

Akhsan Nurramdhan

Machine Learning & AI Enthusiast

LinkedIn:
https://linkedin.com/in/akhsannurramdhan

Portfolio:
https://akhsannurramdhan.site
