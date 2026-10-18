# Dataset Description

Smart Maintenance AI uses three public datasets, one per data modality. The raw data is not stored in this repository. Download each dataset from its original source and place it where the notebook expects it.

| Dataset | Modality | Task in this project | Size |
|---|---|---|---|
| AI4I 2020 Predictive Maintenance | Tabular sensor data | Failure prediction, anomaly detection, clustering | 10,000 rows × 14 columns |
| NASA C-MAPSS (FD001) | Multivariate time series | Remaining Useful Life regression | 20,631 rows × 26 columns |
| NEU-DET | Images | Steel-surface defect classification | 1,800 images, 6 classes |

Maintenance logs are not from a public dataset. The NLP module is zero-shot (no training data) and was tested with hand-written example logs.

---

## 1. AI4I 2020 Predictive Maintenance Dataset

- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/601
- **Type:** Synthetic dataset that reflects real predictive-maintenance data from a milling machine.
- **Records:** 10,000
- **Missing values / duplicates:** none
- **Target:** `Machine failure` (binary)
- **Class balance:** 339 failures (3.39%) vs 9,661 non-failures, so the data is strongly imbalanced.

### Columns

| Column | Description |
|---|---|
| `UDI` | Row identifier (dropped before modelling) |
| `Product ID` | Product identifier (dropped before modelling) |
| `Type` | Product quality variant: L (low), M (medium), H (high). One-hot encoded |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Spindle speed |
| `Torque [Nm]` | Torque (range about 3.8 to 76.6 Nm) |
| `Tool wear [min]` | Cumulative tool wear (range 0 to 253 min) |
| `Machine failure` | **Target.** 1 if the machine failed, else 0 |
| `TWF` | Tool wear failure flag |
| `HDF` | Heat dissipation failure flag |
| `PWF` | Power failure flag |
| `OSF` | Overstrain failure flag |
| `RNF` | Random failure flag |

> **Note on leakage:** `Machine failure` equals 1 whenever any of TWF, HDF, PWF, OSF or RNF is 1. Using these five flags as input features leaks the label, which inflates model scores. The notebook currently includes them, and removing them is planned.

### Preprocessing in the notebook

1. Drop `UDI` and `Product ID`.
2. One-hot encode `Type`.
3. Stratified 80/20 train/test split (8,000 / 2,000 samples, `random_state=42`).
4. Standardise features with `StandardScaler`.

---

## 2. NASA C-MAPSS Turbofan Engine Degradation (subset FD001)

- **Source:** NASA Prognostics Data Repository (Commercial Modular Aero-Propulsion System Simulation).
- **Type:** Simulated run-to-failure data for turbofan engines.
- **FD001:** 100 training engines, one operating condition, one fault mode.
- **Rows used:** 20,631 (one row per engine cycle)

### Columns

| Group | Columns |
|---|---|
| Identifiers | `unit_number` (engine ID), `time_cycles` |
| Operational settings | `setting_1`, `setting_2`, `setting_3` |
| Sensors | `sensor_1` to `sensor_21` |

### Preprocessing in the notebook

1. **RUL label:** not given for training data, so it is computed as (last cycle of the engine) minus (current cycle).
2. **Constant sensors removed:** 19 of 21 sensors have non-zero variance and are kept.
3. **Scaling:** MinMax scaler.
4. **Sliding windows:** 30 cycles × 19 sensors per window, giving 17,631 sequences.
5. **Split:** 14,104 training / 3,527 validation sequences (random split; an engine-wise split would be stricter).

---

## 3. NEU-DET Steel Surface Defect Database

- **Source:** Northeastern University (NEU) surface defect database for hot-rolled steel strips.
- **Type:** Greyscale images of steel surfaces.
- **Images used:** 1,800 in 6 balanced classes (300 per class).
- **Split used:** 240 training and 60 validation images per class (1,440 / 360 total).

### Classes

| Class | Description |
|---|---|
| `crazing` | Network of fine surface cracks |
| `inclusion` | Foreign particles embedded in the surface |
| `patches` | Irregular patch-like surface areas |
| `pitted_surface` | Small pits or cavities |
| `rolled-in_scale` | Scale pressed into the surface during rolling |
| `scratches` | Linear scratch marks |

### Preprocessing in the notebook

1. Resize to 224 × 224.
2. Convert greyscale to 3 channels (ResNet50 input).
3. Augmentation (training only): random horizontal flip and 10° rotation.
4. Normalise with ImageNet statistics.

Expected folder layout after extraction:

```
NEU-DET/
├── train/
│   ├── crazing/
│   ├── inclusion/
│   ├── patches/
│   ├── pitted_surface/
│   ├── rolled-in_scale/
│   └── scratches/
└── validation/
    └── (same six class folders)
```

---

## 4. Maintenance logs (text)

- **Source:** hand-written example logs, no public dataset.
- **Method:** zero-shot classification with `all-MiniLM-L6-v2`. The log is embedded and matched by cosine similarity to ten fault categories: Bearing Failure, Overheating, Excessive Vibration, Lubrication Problem, Motor Failure, Electrical Fault, Gear Failure, Pressure Problem, Leakage, Unknown Fault.
- **Severity:** Low / Medium / High, assigned from a keyword dictionary.
- **Example:** `"excessive vibration and unusual noise"` → Excessive Vibration.

---

## Licences and citation

Each dataset is owned by its original provider. Check and follow the licence and citation requirements on the source pages before redistributing or publishing results.
