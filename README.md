# Automated Multi-Classifier Intrusion Detection System Comparison

## Overview
This repository contains an automated, end-to-end machine learning pipeline designed to process massive network capture datasets (such as CICIDS) and classify malicious network traffic. The project systematically compares three distinct AI architectures—Logistic Regression, Support Vector Machines (SVM), and a Multi-Layer Perceptron (Neural Network)—to identify behavioral signatures of network intrusions. 

The pipeline is engineered for hardware efficiency and fault tolerance, automatically handling data scaling, feature reduction, model balancing, and performance reporting.

## Pipeline Architecture & Modules

The system is broken down into six sequential, highly modularized steps:

### 1. Data Loading & Sampling (`data_loader.py`)
* **Functionality:** Ingests raw, multi-million-row network capture CSV files and scales them down to a computationally manageable footprint.
* **Mechanism:** Utilizes Pandas to apply a strictly enforced 10% stratified sampling strategy, rather than simple random reduction.
* **Rationale:** Bypasses local hardware bottlenecks and reduces training time from days to minutes while perfectly preserving the critical mathematical distribution of rare attacks.

### 2. Data Cleaning & Preprocessing (`data_cleaning.py`)
* **Functionality:** Sanitizes raw data by eliminating `NaN` and Infinity values, converts text labels using `LabelEncoder`, and normalizes mathematical scales using `MinMaxScaler` (binding features between 0 and 1).
* **Mechanism:** Preserves explicit multi-class attack categories (e.g., DDoS, PortScan) to provide granular visibility into model blind spots.
* **Optimization:** Implements a dynamic programmatic filter to automatically scan and drop network features exhibiting over 98% correlation. This eliminates multicollinearity, purging mathematically identical data to improve training speed and algorithm stability without sacrificing predictive power.

*** Before and after the action of dropping redundant data, Heatmap comparison.

<img width="975" height="836" alt="image" src="https://github.com/user-attachments/assets/701065ce-752e-45f9-975d-2c05addebf10" />


After dropping highly correlated data

<img width="975" height="836" alt="image" src="https://github.com/user-attachments/assets/fc183a7c-6ed0-4084-9cc7-20a22198e893" />

### 3. Exploratory Data Analysis (`eda_engine.py`)
* **Functionality:** Computes Pearson correlation coefficients to uncover and visualize the statistical "behavioral signatures" distinguishing malicious traffic from normal operations.
* **Mechanism:** Uses Seaborn and Matplotlib to render targeted visual intelligence.
* **Optimization:** Bypasses unreadable 80x80 correlation matrices by dynamically isolating and displaying only the top 15 features intersecting directly with the target label (e.g., highlighting `Fwd IAT Max` for timing attacks).

### 4. Pipeline Orchestration (`main.py`)
* **Functionality:** Serves as the central command hub, directing workflow across modules and splicing data into a model-training-ready format.
* **Mechanism:** Partitions data into an isolated 70/30 training and testing split using Scikit-Learn, strictly enforcing target stratification (`stratify=y`) to maintain rare attack ratios.
* **Safety Switches:** Integrates dynamic safety filters to automatically bypass training on single-class baseline days and drop unlearnable minority classes (fewer than 2 instances), preventing pipeline crashes and ensuring mathematically viable datasets.

### 5. Model Design & Training (`trainer.py`)
* **Functionality:** Trains three distinct AI architectures to recognize intrusion signatures: Logistic Regression (linear baseline), SVM (complex non-linear boundaries), and a Multi-Layer Perceptron (Neural Network).
* **Mechanism:** Forces algorithmic balancing (`class_weight='balanced'`) for LR and SVM to prioritize minority threat classes rather than lazily memorizing the overwhelmingly benign majority traffic.
* **Optimization:** Activates early stopping for the Neural Network and explicitly extracts probability arrays for the SVM to enable downstream statistical confidence mapping.

### 6. Performance Evaluation & Metrics (`evaluator.py`)
* **Functionality:** Grades predictive accuracy against a hidden testing set and generates professional reporting artifacts.
* **Mechanism:** Calculates Precision, Recall, F1-scores, and AUC metrics. Multi-class probabilities are mathematically converted into a unified "Benign vs. Malicious" binary array to generate ROC overlay charts.
* **Reporting:** Features dynamic label mapping to prevent index crashes on dropped rare classes, alongside an automated Excel exporter for the final metrics to eliminate manual entry errors.

## Getting Started

**1. Set up the Virtual Environment:**
```bash
python -m venv venv
```
2. Activate the Environment:

Windows (PowerShell): 
```bash
.\venv\Scripts\activate
```

Mac/Linux: 
```bash
source venv/bin/activate
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Execute the Pipeline:
```bash
python main.py
```
