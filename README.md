Loan Approval Prediction Web App (Flask + ML)

## Overview

This project is a Flask web application that predicts loan approval based on user-provided information. It serves a trained machine learning model and includes a separate training script to build and evaluate traditional ML ensembles.

## Features

- End-to-end loan approval prediction web UI built with Flask
- Robust preprocessing with `RobustScaler` and categorical encoders saved via `joblib`
- PyTorch inference model (`model.pth`) with advanced architecture
- Optional traditional ML ensemble training (`train_ensemble.py`)
- Clear results page showing predicted decision and confidence
- Defensive input validation with helpful error messages

## Tech Stack

- Python, Flask
- PyTorch (inference model loaded from `model.pth`)
- scikit-learn (preprocessing + alternative ensemble model in `train_ensemble.py`)
- numpy, pandas, joblib, matplotlib

## Repository Structure

```
.
├── app.py                      # Flask app entrypoint
├── train_ensemble.py           # Training script for traditional ML ensemble
├── LAP.csv                     # Dataset used by the training script
├── model.pth                   # PyTorch model used by the Flask app (inference)
├── scaler.pkl                  # RobustScaler used for preprocessing
├── label_encoders.pkl          # Column-wise LabelEncoders
├── label_encoder_target.pkl    # LabelEncoder for the target variable
├── templates/
│   ├── index.html              # Form for collecting user input
│   └── results.html            # Results page
└── user_input.py               # (If used) helper for input handling
```

## Code Overview

- `app.py`: Flask routes, input validation, feature engineering, categorical encoding, scaling, PyTorch model loading, and prediction logic. Produces a confidence score for classification.
- `train_ensemble.py`: Loads `LAP.csv`, performs extensive feature engineering, trains multiple scikit-learn models, evaluates, and saves preprocessing artifacts and an ensemble checkpoint.
- `templates/`: HTML templates (`index.html`, `results.html`) for the web UI.
- `model.pth`: PyTorch model state dict used by the web app.
- `scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`: Preprocessing artifacts saved with `joblib`.

## Project Description

This application demonstrates a practical ML workflow: a trained model is packaged with its preprocessing artifacts and served via a simple Flask interface. The app collects core applicant details, derives additional predictive features, applies encoders/scaling, and performs a prediction with a confidence score. The result is presented as an eligibility decision (Yes/No) along with a confidence threshold to avoid overconfident outputs on uncertain inputs.

The repository also includes a separate training script (`train_ensemble.py`) that builds classic ML baselines (RandomForest, GradientBoosting, LogisticRegression) and an ensemble for comparison and experimentation.

## Inputs Collected

- Loan_ID, Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
- Credit_History, Property_Area

Derived features include Total Income, Income-to-Loan ratio, Loan-to-Income ratio, and categorical bins for income/loan levels.

## Quickstart

1) Create a virtual environment and activate it

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Training (optional)

If you want to retrain preprocessing and an ensemble model (RandomForest, GradientBoosting, LogisticRegression), run:

```bash
python train_ensemble.py
```

This will output updated artifacts like `scaler.pkl`, `label_encoders.pkl`, and `label_encoder_target.pkl`. The script also saves `ensemble_model.pkl` for reference; the Flask app loads the PyTorch model from `model.pth` for inference.

## How it works

1) User submits the form at `/` (`templates/index.html`).
2) `app.py` validates inputs, engineers additional features, encodes categoricals, and scales numerics.
3) The preprocessed vector is passed to the PyTorch model loaded from `model.pth`.
4) The output is interpreted (classification with confidence) and shown on `/results`.

### Routes

- `/`: Home page with input form
- `/results`: Results page with prediction and confidence

## Deployment

- Production WSGI (example):
  - `pip install gunicorn waitress`
  - Gunicorn (Linux): `gunicorn -w 2 -b 0.0.0.0:8000 app:app`
  - Waitress (Windows): `waitress-serve --listen=0.0.0.0:8000 app:app`
- Cloud (Render/Heroku):
  - Set Python version and install from `requirements.txt`.
  - Ensure artifacts (`model.pth`, `scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`) are deployed.
  - Set environment variable `FLASK_SECRET` (optional) for session security.

## Environment Variables

- `FLASK_SECRET`: Optional; if set, overrides `app.secret_key` in `app.py`.

## Notes

- The Flask app expects the preprocessing artifacts (`scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`) to be present in the project root.
- The dataset `LAP.csv` is used by the training script only. The web app does not require it at runtime.
- To replace the inference model, update `model.pth` and ensure its input feature ordering matches the preprocessing pipeline.

## How to publish on GitHub

1) Initialize git (inside the project folder):

```bash
git init
git add .
git commit -m "Initial commit: Loan approval prediction app"
```

2) Create a new empty repository on GitHub (no README/.gitignore). Then connect and push:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.


