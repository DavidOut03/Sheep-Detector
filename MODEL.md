# 🧠 Model Documentation

## 📌 Introduction

The goal of this project is to detect **sheep in infrared images** captured by drones. This is a challenging task due to:
- Small size and low contrast of sheep in thermal imagery
- Variable environmental conditions
- Limited dataset size (~176 images, mixed with and without sheep, most without sheep)


## 🔎 Model Search & Selection

### Why Object Detection?
Since we need not only to confirm the presence of sheep but also locate them in each image, object detection models are ideal. Our requirements:
- High **accuracy** with small and low-res objects
- Generalize well with a limited dataset
- Robust under diverse outdoor conditions

### Literature Review

Our model selection was guided by recent research and public implementations:
- [YOLO for Aerial Detection](https://www.mdpi.com/1424-8220/23/14/6423)
- [YOLOv8 on Infrared Datasets](https://www.mdpi.com/2504-446X/8/9/479)
- [YOLO for Wildlife in IR](https://arxiv.org/abs/2409.06259)

### YOLO Family

| Model      | Pros                                                      | Notes |
|------------|-----------------------------------------------------------|-------|
| YOLOv5     | Lightweight, proven, widely used                          | Used in EL-YOLO aerial image studies |
| YOLOv8     | Better accuracy, great with small objects, IR support     | Chosen for this PoC |
| YOLOv11    | Newest release, promising performance                     | Candidate for future testing |

**Chosen Model:** `YOLOv11` (Ultralytics)

It provides the best balance between detection performance and ease of integration. Our main focus is high accuracy for small object detection under low-resolution constraints.


## ⚙️ Hyperparameters

We plan to tune the following:

| Parameter              | Description                                                              | Strategy         | Typical Range           |
|-------------------------|--------------------------------------------------------------------------|------------------|--------------------------|
| `learning_rate` (`lr0`) | Initial learning rate controlling step size during training.             | Random Search    | 1e-5 to 1e-1             |
| `momentum`              | Momentum factor to accelerate optimizer convergence.                    | Random Search    | 0.6 to 0.98              |
| `weight_decay`          | Regularization term to avoid overfitting by penalizing large weights.    | Random Search    | 0.0 to 0.001             |
| `warmup_epochs`         | Gradual learning rate increase phase during early training.              | Random Search    | 0 to 5                   |
| `box`                   | Loss weight for object localization accuracy.                           | Random Search    | 0.02 to 0.2              |
| `cls`                   | Loss weight for object classification accuracy.                         | Random Search    | 0.2 to 4.0               |




## 📊 Evaluation Metrics

We will use the following metrics to evaluate model performance:

| Metric         | Purpose                                             |
|----------------|-----------------------------------------------------|
| `Precision`    | Measures correctness of sheep detections            |
| `Recall`       | Measures how many sheep were successfully detected  |
| `F1 Score`     | Balances precision and recall                       |
| `IoU (avg)`    | Measures how well predicted boxes align with ground truth |

These metrics help us understand how well the model handles small object detection and overlapping cases.


## ✅ Summary

- **Model:** YOLOv11 (Ultralytics)
- **Focus:** High-accuracy detection of sheep in thermal drone imagery
- **Tuning:** Batch size, learning rate, frozen layers, confidence threshold
- **Evaluation:** Precision, recall, F1 score, and average IoU


> This is an evolving proof-of-concept. 
