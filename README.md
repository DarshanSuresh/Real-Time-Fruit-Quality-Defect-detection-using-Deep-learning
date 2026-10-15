# Real-Time Fruit Quality & Defect Detection Using Deep Learning

<p align="center">
  <strong>A Comparative Study of YOLO, EfficientNet, DenseNet, and YOLO–EfficientNet Hybrid Models</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Deep%20Learning-blue" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Analysis-green" />
  <img src="https://img.shields.io/badge/Task-Defect%20Detection-orange" />
  <img src="https://img.shields.io/badge/Status-Research%20%26%20Development-yellow" />
</p>

## Project Overview

**Real-Time Fruit Quality & Defect Detection Using Deep Learning** is a computer vision research project that aims to automatically identify fruit defects, estimate defect severity, and classify fruit quality grades using camera-based images.

Traditional fruit inspection often depends on manual observation, which can be time-consuming and subject to human error. This project investigates deep learning techniques to support faster, more consistent, and automated fruit quality assessment.

The study evaluates four model configurations: YOLO, EfficientNet, DenseNet, and a proposed YOLO–EfficientNet hybrid approach. Each model will first be evaluated in its original pretrained configuration and then fine-tuned on the selected fruit image dataset. Their performance will be compared under different dataset splits, batch sizes, and training epochs.

The objective is to identify a suitable model configuration that balances detection performance, classification accuracy, inference speed, and computational requirements for real-time applications.

## Research Objectives

- Detect visible fruit defects from camera-based images.
- Identify defect categories such as bruising, rot, discoloration, cuts, and surface damage, depending on dataset availability.
- Investigate defect severity and maturity assessment.
- Assign quality grades based on defined grading criteria.
- Compare original pretrained models with fine-tuned models.
- Evaluate different training and validation splits: 80:20, 70:30, and 60:40.
- Investigate the effect of batch sizes 8, 16, and 32.
- Measure accuracy, precision, recall, F1-score, execution time, and inference performance.
- Investigate the feasibility of real-time deployment in agricultural inspection and post-harvest handling.

## Models Under Investigation

| Model | Primary Role | Research Focus |
|---|---|---|
| YOLO | Object detection | Fruit localization, defect localization, and detection speed |
| EfficientNet | Image classification | Fruit condition classification and quality grading |
| DenseNet | Image classification | Feature reuse and recognition of visual defect patterns |
| YOLO + EfficientNet | Hybrid approach | Combining localization with detailed image classification |

**Important:** The exact YOLO version, EfficientNet variant, and DenseNet variant will be documented after implementation. The hybrid architecture will be defined explicitly so that its design and contribution can be reproduced.

## Research Workflow

1. **Dataset Collection:** Collect and organize fruit images from a camera-based image acquisition process or a suitable public dataset.
2. **Data Annotation:** Label fruit classes, defect locations, defect categories, and severity levels where supported by the data.
3. **Preprocessing:** Resize images, normalize pixel values, and apply suitable data augmentation to training data only.
4. **Baseline Evaluation:** Evaluate original pretrained models using a consistent test protocol.
5. **Fine-Tuning:** Adapt pretrained weights to the selected fruit dataset.
6. **Experimental Evaluation:** Train and evaluate models using different dataset splits, batch sizes, and epoch configurations.
7. **Performance Comparison:** Compare baseline and fine-tuned models using consistent evaluation metrics.
8. **Real-Time Testing:** Measure inference latency and throughput on the selected hardware.
9. **Quality Grading:** Evaluate whether predicted defects can support predefined fruit quality grades.

## Proposed Methodology

The proposed system follows a modular pipeline:

```text
Camera / Image Dataset
          |
          v
   Image Acquisition
          |
          v
 Preprocessing & Annotation
          |
          v
  Dataset Split (Train/Val/Test)
          |
          v
  +-------+--------+----------+
  |                |          |
 YOLO          EfficientNet  DenseNet
  |                |          |
  +-------+--------+----------+
          |
          v
   Fine-Tuning Experiments
          |
          v
 YOLO + EfficientNet Hybrid
          |
          v
 Defect Detection & Classification
          |
          v
 Defect Severity / Maturity Assessment
          |
          v
     Quality Grading
          |
          v
 Metrics, Graphs & Model Comparison
```

The final architecture will depend on the selected YOLO implementation, dataset annotations, and hybrid integration strategy.

## Experimental Design

### 1. Dataset Split Comparison

| Experiment | Training Data | Validation/Test Allocation |
|---|---:|---:|
| Split A | 80% | 20% |
| Split B | 70% | 30% |
| Split C | 60% | 40% |

The final implementation will specify validation and test allocations separately. Where possible, a fixed, independent test set will be retained to ensure a fair comparison between models.

### 2. Batch Size Comparison

| Batch Size | Purpose |
|---|---|
| 8 | Evaluate training with smaller batches |
| 16 | Evaluate a medium batch configuration |
| 32 | Evaluate a larger batch configuration |

### 3. Baseline vs. Fine-Tuned Models

For each model, the study will document:

- Original pretrained model performance.
- Fine-tuned model performance.
- Change in accuracy and other evaluation metrics.
- Training duration and inference time.
- Model size and computational requirements.
- Changes in defect recognition and quality grading performance.

### 4. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives identified |
| F1-score | Harmonic mean of precision and recall |
| mAP@0.5 | Object detection performance at IoU 0.5 |
| mAP@0.5:0.95 | Object detection performance across IoU thresholds |
| Inference Time | Time taken to process an image |
| FPS | Images or frames processed per second |

Classification metrics will be used for classification tasks, while detection metrics will be used for object detection. Results will be reported according to the task performed by each model.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch and/or Ultralytics |
| Computer Vision | OpenCV |
| Model Architectures | YOLO, EfficientNet, DenseNet |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn and task-specific detection metrics |
| Development | Jupyter Notebook / Google Colab / VS Code |
| Version Control | Git and GitHub |

The final stack will reflect the libraries and hardware actually used in the implementation.

## Suggested Repository Structure

```text
Real-Time-Fruit-Quality-Detection/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_yolo_baseline.ipynb
│   ├── 03_efficientnet_baseline.ipynb
│   ├── 04_densenet_baseline.ipynb
│   ├── 05_model_finetuning.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_yolo.py
│   ├── train_efficientnet.py
│   ├── train_densenet.py
│   ├── hybrid_model.py
│   ├── evaluate.py
│   └── inference.py
│
├── configs/
│   └── experiment_config.yaml
│
├── results/
│   ├── metrics/
│   ├── graphs/
│   └── tables/
│
├── poster/
│   └── research_poster.pdf
│
└── paper/
    └── ieee_manuscript.pdf
```

*This is the proposed structure; files and directories will be added as the project develops.*

## Expected Research Outputs

The research will aim to produce:

- A comparative analysis of four model configurations.
- Before-and-after fine-tuning performance tables.
- Accuracy, precision, recall, and F1-score graphs.
- Object detection evaluation results where applicable.
- Execution-time and inference-speed comparisons.
- An ablation study of relevant training configurations.
- A proposed methodology for defect severity assessment and quality grading.
- A six-page IEEE-style research manuscript.
- A research poster summarizing the methodology and findings.

All numerical results will be generated from actual experiments rather than assumed in advance.

## Research Gap

The study will investigate whether combining fast object detection with a specialized image classification network can improve fruit defect assessment without imposing unacceptable inference-time overhead.

The research will focus on:

1. Comparing pretrained baselines against dataset-specific fine-tuning.
2. Evaluating detection and classification approaches using appropriate metrics.
3. Investigating defect severity and maturity grading in addition to defect presence.
4. Measuring the trade-off between accuracy and real-time processing speed.
5. Assessing the generalization of the models across different fruit appearances and imaging conditions.

These research gaps will be refined after reviewing relevant deep learning literature and identifying limitations in existing studies.

## Potential Applications

- Automated fruit sorting and grading.
- Post-harvest quality inspection.
- Agricultural supply chains.
- Food processing and packaging.
- Smart farming and machine vision systems.
- Camera-assisted quality control.

## Current Limitations

- Performance depends on dataset diversity, image quality, and annotation accuracy.
- Surface defects may be difficult to identify under poor lighting or occlusion.
- Defect severity and maturity require clearly defined labels and grading criteria.
- Real-time performance depends on the deployment hardware and input resolution.
- A hybrid architecture may introduce additional computational overhead.
- Results obtained on one fruit type or dataset may not generalize to others.

## Future Scope

- Multi-fruit detection and grading.
- Camera-stream processing with real-time alerts.
- Edge deployment on embedded systems.
- Explainable AI for visualizing defect-related regions.
- Improved severity estimation using segmentation or localization.
- Evaluation under different lighting and background conditions.
- Integration with automated sorting systems.

## Research Paper Plan

The accompanying IEEE-style manuscript will include:

1. Introduction
2. Literature Review
3. Research Gap and Objectives
4. Proposed Methodology
5. Model Architectures and Experimental Setup
6. Results and Comparative Analysis
7. Ablation Study, Limitations, and Future Scope
8. Conclusion
9. References

The literature review will include approximately 16–20 relevant deep learning research papers. A research poster will communicate the problem, methodology, architecture, and experimentally verified results.

## Project Status

**Current Phase:** Project Planning and Literature Review

Upcoming milestones:

- [ ] Identify and review 16–20 deep learning research papers.
- [ ] Select the fruit dataset and define its annotation scheme.
- [ ] Establish original pretrained model baselines.
- [ ] Fine-tune YOLO, EfficientNet, and DenseNet.
- [ ] Design and implement the hybrid architecture.
- [ ] Run controlled experimental comparisons.
- [ ] Prepare results, IEEE manuscript, and research poster.

## Reproducibility and Research Integrity

The project will document dataset sources, model versions, hyperparameters, train/validation/test splits, hardware specifications, and evaluation procedures. Results will be reported transparently, including unsuccessful experiments and limitations where relevant.

The manuscript and poster will be written with appropriate citations, original analysis, and proper attribution to the authors of existing work. Any similarity or AI-detection requirements will be addressed through genuine original writing, verified references, and transparent research practices rather than claims of guaranteed detector scores.

## License

A suitable open-source license will be selected before public release. Dataset licenses and third-party model licenses will be respected separately.

---

**Project Title:** Real-Time Fruit Quality & Defect Detection Using Deep Learning

**Research Focus:** Defect detection, defect severity, maturity assessment, and fruit quality grading.

**Core Comparison:** YOLO vs. EfficientNet vs. DenseNet vs. YOLO–EfficientNet Hybrid.
