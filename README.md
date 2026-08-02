# 🏏 Cricket Match Outcome Predictor

A machine learning pipeline that predicts **T20 cricket match outcomes** using historical match and player data, deployed as an interactive **Streamlit** web app for real-time predictions.

## 📌 Overview

This project builds an end-to-end ML pipeline — from raw match data to a deployed prediction app — to forecast T20 cricket match results based on in-game and historical statistics.

## ✨ Features

- Predicts match outcomes using historical T20 match and player-level data
- Feature-engineered pipeline for improved model accuracy
- Trained and compared multiple models (**XGBoost**, **CatBoost**) to select the best performer
- Real-time, interactive predictions via a **Streamlit** web app

## 🛠️ Tech Stack

- **Language:** Python
- **ML Libraries:** XGBoost, CatBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Web App:** Streamlit

## 📂 Repository Structure

```
Cricket_match_predictor/
├── app.py                   # Streamlit app for real-time prediction
├── preprocessing.ipynb      # Data cleaning & feature engineering
├── cricket_model.ipynb      # Model training & evaluation
├── xgboost_.pkl              # Trained XGBoost model
├── catboost_.pkl              # Trained CatBoost model
├── requirements.txt          # Project dependencies
└── .devcontainer/            # Dev container configuration
```

## ⚙️ How It Works

1. **Preprocessing** (`preprocessing.ipynb`): Cleans raw match data and engineers predictive features from historical match and player statistics.
2. **Model Training** (`cricket_model.ipynb`): Trains and tunes classifiers (XGBoost, CatBoost) to predict match outcomes, comparing performance to select the best model.
3. **Deployment** (`app.py`): Serves the trained model through a Streamlit interface, allowing users to input match parameters and get real-time outcome predictions.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/pushti0102/Cricket_match_predictor.git
cd Cricket_match_predictor
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

## 📊 Model

The final prediction model is selected from trained **XGBoost** and **CatBoost** classifiers based on evaluation metrics on historical T20 match data.

## 📬 Contact

For questions or collaboration, reach out via [GitHub](https://github.com/pushti0102).

---

*Built as part of exploring applied machine learning in sports analytics.*
