Loan Approval Prediction Web App (Flask + ML)

## Overview

This project is a Flask web application that predicts loan approval based on user-provided information. It serves a trained machine learning model and includes a separate training script to build and evaluate traditional ML ensembles.

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

You can add a license of your choice (e.g., MIT). If you are unsure, MIT is a simple permissive option.


