# Data-Driven Predictive Maintenance Platform for Remaining Useful Life (RUL) Prediction of Industrial Equipment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Benchmark: NASA PHM08](https://img.shields.io/badge/Benchmark-NASA%20PHM08-orange.svg)](https://data.nasa.gov/)

Academic prototype developed as part of **Project-Based Learning (PBL)** in **Artificial Intelligence & Machine Learning**, Thakur College of Engineering & Technology (TCET), Mumbai (Academic Year 2026-27).

**Project Authors:**
* Yashavi Kharwar (Roll No: 07)
* Nikita Mishra (Roll No: 24)
* Satyam Mishra (Roll No: 25)

**Under the Guidance of:**
* Ms. Seema Nagrani (Assistant Professor of AI&ML, TCET Mumbai)

---

## 1. Project Overview & Objective

Unexpected failure of complex industrial assets like aircraft turbofan engines leads to severe operational risks, downtime, and massive economic losses. This platform develops a data-driven **Prognostics and Health Management (PHM)** system to predict the **Remaining Useful Life (RUL)** of turbofan engines using the **NASA PHM08 challenge dataset**.

$$\text{RUL}(t) = T_f - t$$

Where $T_f$ is the failure cycle and $t$ is the current operating cycle.

### Key Objectives & Dashboard Modules
* **RO1 (Data Analysis):** Analyze the PHM08 multivariate time-series across 6 distinct flight conditions and identify equipment wear signatures.
* **RO2 (Model Development):** Develop both a **Baseline Regression Model (Random Forest)** and a **Temporal Deep Learning Model (LSTM)** that captures long-term degradation dependencies.
* **RO3 (Quantitative Evaluation):** Benchmark performance using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and the **Asymmetric PHM08 Challenge Score** under strict, leakage-free engine-wise separation.
* **RO4 (Decision-Support Dashboard):** Deploy an industrial-themed Streamlit dashboard with 8 dedicated modules:
  1. **Overview Dashboard:** Fleet metrics (Total engines, Healthy, Warning, Critical), live distribution donut chart, average RUL.
  2. **RUL Prediction:** Engine selector, model toggle, glowing KPI cards, RUL degradation curve, sensor telemetry overlay, and natural-language diagnostic explanation.
  3. **Sensor Analytics:** 21 sensor channels + 3 operational settings, multi-sensor comparison, cross-engine fleet degradation envelopes, and correlation heatmap.
  4. **Model Performance:** Empirical evaluation metrics (MAE, RMSE, PHM08 score), actual vs predicted scatter plot, residual error histogram, and LSTM loss convergence curves.
  5. **Dataset Explorer:** Search and filter raw engine telemetry, column descriptions, and 100% data integrity audit.
  6. **Maintenance Insights:** Priority Action Matrix (P1 Immediate, P2 Scheduled, P3 Routine) ranking engines by lowest RUL with suggested inspection protocols.
  7. **Model Training:** Interactive training studio with configurable hyperparameters (epochs, batch size, window $L$, RUL cap) and live progress tracking.
  8. **About the Project:** Presentation showcase for evaluators detailing team members, guide, mathematical equations, and Viksit Bharat / SDG alignment.

---

## 2. System Architecture & Methodology

```
┌─────────────────────────────────┐
│     NASA PHM08 Benchmark       │ 218 train run-to-failure engines
│ (train.txt, test.txt, final.txt)│ 218 test engines ending before failure
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Data & Feature Layer       │ - Check missing values and constant channels
│ (Data Quality, RUL Generator)   │ - Generate piecewise linear target (cap=125)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Zero Data Leakage Splitting   │ - 152 engines (70%) -> Training
│   (Engine-Wise Partitioning)    │ - 32 engines (15%)  -> Validation (Early Stopping)
└────────────────┬────────────────┘ - 34 engines (15%)  -> Held-out Test Evaluation
                 │
                 ▼
┌─────────────────────────────────┐
│  Standardization & Sequencing   │ - StandardScaler fitted ONLY on training engines
│    (Sliding Window L = 30)      │ - Input shape: (N, 30 cycles, 24 features)
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐ ┌───────────────┐
│ Model A (RF)  │ │ Model B(LSTM) │ 64-unit LSTM + Dropout(0.2) + Dense(32, ReLU)
│ Baseline Reg. │ │ Sequence DL   │ Adam optimizer + Early stopping
└───────┬───────┘ └───────┬───────┘
        │                 │
        └────────┬────────┘
                 ▼
┌─────────────────────────────────┐
│     Model Evaluation Layer      │ - MAE, RMSE, R-squared
│    (Asymmetric PHM08 Score)     │ - Cycle-level & Trajectory-end evaluations
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Streamlit Decision Dashboard   │ - Real-time RUL prediction & health indicators
│    (app.py - Industrial UI)     │ - Degradation curve visualization & sensor trends
└─────────────────────────────────┘
```

---

## 3. Technology Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-learn, Joblib
* **Deep Learning:** Recurrent LSTM sequence model (Long Short-Term Memory neural network for temporal degradation modeling)
* **Visualization:** Plotly (interactive), Matplotlib (static)
* **Web Dashboard:** Streamlit

---

## 4. Asymmetric PHM08 Challenge Score

In aerospace prognostics, the consequence of late predictions is asymmetric. An early prediction causes slightly premature maintenance, whereas an unexpected in-flight failure causes catastrophic risk. The challenge scoring function is formulated as:

$$d_i = \hat{y}_i - y_i$$

$$S = \sum_{d_i < 0} \left( e^{-d_i / 13} - 1 \right) + \sum_{d_i \ge 0} \left( e^{d_i / 10} - 1 \right)$$

* **Late predictions ($d_i \ge 0$):** Heavily penalized with exponential factor $e^{d/10}$.
* **Early predictions ($d_i < 0$):** Less severely penalized with exponential factor $e^{-d/13}$.

---

## 5. Empirical Benchmark Results (34 Held-Out Engines)

The models were evaluated under strict zero-leakage conditions on the **34 held-out test engines**:

### Cycle-Level Evaluation (5,929 Temporal Windows)
| Model | MAE (Cycles) | RMSE (Cycles) | R² Score | PHM08 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Model A: Random Forest Baseline** | 16.29 | 20.71 | 0.754 | 119,229.3 |
| **Model B: Recurrent LSTM Network** | **14.67** | **20.24** | **0.765** | **117,908.4** |

### Trajectory-End Evaluation (Final Inspection Point Before Failure)
| Model | MAE (Cycles) | RMSE (Cycles) | Asymmetric PHM08 Score |
| :--- | :---: | :---: | :---: |
| **Model A: Random Forest Baseline** | 8.41 | 9.13 | 50.60 |
| **Model B: Recurrent LSTM Network** | **7.65** | **8.32** | **43.82** *(13.4% lower risk penalty)* |

---

## 6. Project Directory Structure

```
RUL Prediction/
├── app.py                      # Interactive Streamlit industrial cockpit
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation and project manual
├── .gitignore                  # Git exclusion rules
├── .streamlit/
│   └── config.toml             # Streamlit configuration & dark theme definition
├── scripts/
│   └── download_data.py        # Automated NASA PHM08 dataset retriever
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Constants, paths, and sensor descriptions
│   ├── data_loader.py          # Data ingestion and quality audit
│   ├── preprocessing.py        # Engine-wise split, scaling, sliding sequences
│   ├── feature_engineering.py  # Variance analysis and feature selection
│   ├── models.py               # Random Forest Baseline & LSTM Regressor
│   ├── train.py                # End-to-end training and evaluation pipeline
│   ├── evaluate.py             # MAE, RMSE, and asymmetric PHM08 score
│   ├── predict.py              # Inference and trajectory prediction
│   └── visualizations.py       # Plotly and Matplotlib charting functions
├── notebooks/
│   └── pbl_rul_experiments.ipynb # Jupyter/Colab notebook with complete workflow
├── tests/
│   └── test_pipeline.py        # Automated unit and integration test suite
├── data/                       # Dataset directory (train.txt, test.txt, final_test.txt)
└── models/                     # Saved artifacts (scaler, weights, metadata, metrics)
```

---

## 7. Installation & Execution Guide

### Step 1: Clone and Set Up Environment
```powershell
# Open terminal in project directory
cd "C:\Users\Nikita\Desktop\RUL Prediction"

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset
To automatically download and unpack the NASA PHM08 challenge files (`train.txt`, `test.txt`, `final_test.txt`):
```powershell
python scripts/download_data.py
```

### Step 3: Run Automated Test Suite
Verify that all components (data loader, zero-leakage split, scaling, models, metrics) pass:
```powershell
python -m unittest tests/test_pipeline.py
```

### Step 4: Train Models
Run the training pipeline. This trains both the Random Forest baseline and the LSTM neural network on the training engines and evaluates them on held-out test engines:
```powershell
python -m src.train --epochs 20 --batch-size 64 --window 30 --cap-rul 125
```
*Trained artifacts (`scaler.joblib`, `baseline_rf.joblib`, `lstm_model.joblib`, `metadata.json`, `metrics_summary.json`) are automatically saved to `models/`.*

### Step 5: Launch the Interactive Dashboard
Launch the Streamlit predictive maintenance dashboard:
```powershell
python -m streamlit run app.py
# or: streamlit run app.py
```
Then open your browser to `http://localhost:8501`.

---

## 8. Google Colab GPU Training (Optional)

To run the experiments on Google Colab with GPU acceleration:
1. Open Google Colab and upload `notebooks/pbl_rul_experiments.ipynb`.
2. In Colab, navigate to **Runtime > Change runtime type** and select **T4 GPU**.
3. Run all cells sequentially. The notebook will automatically download the dataset and execute training and evaluation.

---

## 9. Academic Prototype Disclaimer

This software is an academic demonstration developed for Project-Based Learning in Artificial Intelligence & Machine Learning at Thakur College of Engineering & Technology (TCET), Mumbai. It is designed to assist maintenance planning and demonstrate prognostic modeling. It is **not** a certified aircraft engine maintenance system and must not be used for safety-critical aviation control.
