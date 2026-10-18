# Smart Maintenance AI

**Explainable Multimodal AI for Predictive Maintenance and Root-Cause Analysis in Smart Manufacturing**

Smart Maintenance AI combines four kinds of industrial evidence (sensor readings, run-to-failure time series, component images and operator maintenance logs) into one explainable machine-health report. It predicts failure risk, estimates Remaining Useful Life (RUL), detects anomalies, classifies surface defects, interprets free-text logs, and fuses everything into an overall status with maintenance recommendations.


---

## Features

| Module | Data | Models | Output |
|---|---|---|---|
| Failure prediction | AI4I 2020 (tabular) | Logistic Regression, Decision Tree, Random Forest, **XGBoost** | Failure probability, risk level |
| Anomaly detection | AI4I 2020 | Isolation Forest, K-Means (k=3) + PCA | Anomaly flag, operating state |
| RUL estimation | NASA C-MAPSS FD001 | **LSTM** (PyTorch) | Remaining cycles |
| Defect classification | NEU-DET (images) | **ResNet50** (transfer learning) + Grad-CAM | Defect class, heat-map |
| Log analysis | Free-text maintenance logs | `all-MiniLM-L6-v2` (zero-shot) | Fault type, severity, keywords |
| Multimodal fusion | Outputs of all modules | Weighted risk score | Status: Healthy / Moderate / High Risk / Critical, plus recommendations |
| Explainability | - | SHAP, Grad-CAM | Root-cause factors, visual evidence |

## Workflow

```
Industrial data
 ├── Sensor data ──► ML / DL (XGBoost, LSTM) ─┐
 ├── Images ───────► CV (ResNet50 + Grad-CAM) ─┼─► Multimodal fusion ─► Failure / Fault type / RUL
 └── Text logs ────► NLP (MiniLM embeddings) ──┘                              │
                                                  Root-cause analysis ◄───────┘
                                                          │
                                           Explainable AI (SHAP) ─► Maintenance recommendation
```

### Fusion logic

Each module is converted to a 0-1 risk and combined:

| Source | Weight |
|---|---|
| Sensor failure probability | 0.35 |
| RUL (banded: ≤10 cycles = 1.0, ≤25 = 0.75, ≤50 = 0.5, ≤100 = 0.25, else 0) | 0.30 |
| Image defect risk × confidence | 0.20 |
| Log severity (Low 0.25, Medium 0.60, High 1.0) | 0.15 |

Status thresholds: risk ≥ 0.75 → **Critical**, ≥ 0.50 → **High Risk**, ≥ 0.25 → **Moderate Risk**, otherwise **Healthy**.

## Results

| Task | Result |
|---|---|
| Failure prediction (XGBoost, 2,000 test samples) | Accuracy 0.999, F1 0.985, ROC-AUC 0.996 |
| RUL (LSTM, C-MAPSS FD001) | MAE 25.07 cycles, RMSE 38.81 cycles |
| Defect classification (ResNet50, 360 validation images) | Accuracy 91.4%, F1 0.911 |
| Anomaly detection (Isolation Forest) | 98 of 2,000 test samples flagged (~5%) |
| Log analysis | Correct fault identified on test logs (qualitative; no labelled set) |

## Datasets

Three public datasets are used. They are **not** included in this repository. See [DATASET.md](DATASET.md) for a full description, columns, sizes and download instructions.

| Dataset | Modality |
|---|---|
| AI4I 2020 Predictive Maintenance | Tabular sensor data (10,000 rows) |
| NASA C-MAPSS (FD001) | Multivariate time series (20,631 rows) |
| NEU-DET | Steel-surface images (1,800 images, 6 classes) |

## Repository structure

```
.
├── Smart_Maintenance_AI.ipynb   # Full Colab notebook (training, evaluation, fusion)
├── index.html    # Interactive web demo and graph gallery
├── README.md
├── DATASET.md
├── Smart_Maintenance_AI-Project Report

```


## Future work

- Retrain the sensor model without failure-flag features and predict failure *type*.
- Evaluate the LSTM on engine-wise and official test splits, with RUL capping.
- Fine-tune deeper ResNet50 layers and calibrate confidences.
- Learn or calibrate fusion weights on data where all modalities come from the same machine.
- Extract components and maintenance history from logs.

## References

- A Systematic Literature Review of Machine Learning Methods Applied to Predictive Maintenance (2019)
- Smart Fault Diagnosis with Deep Learning for Predictive Maintenance
- Interpretable Fault Detection Approach With Deep Neural Networks to Industrial Applications
- Deep, Generative, and Explainable Models for Anomaly Detection in Smart Manufacturing
- Explainable Predictive Maintenance of Rotating Machines Using LIME, SHAP, PDP, ICE, IEEE Access, vol. 12, 2024
- Explainable Artificial Intelligence for Predictive Maintenance Applications, 2020


