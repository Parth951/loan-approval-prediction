# Code Overview

## Data Flow

1. Request hits `/` (GET) → renders `templates/index.html` form
2. User submits form (POST) → `app.py` collects raw values into `user_input`
3. `validate_and_preprocess_input` enforces types/ranges and normalizes strings
4. `create_advanced_features` adds derived features (totals, ratios, bins)
5. Convert to DataFrame → encode categoricals via `label_encoders.pkl`
6. Scale features via `scaler.pkl` (RobustScaler)
7. Load PyTorch model `model.pth` and run inference
8. Map class index back using `label_encoder_target.pkl` (for classification)
9. Store `prediction`, `confidence`, `user_input` in session and redirect to `/results`

## Key Functions (app.py)

- `validate_and_preprocess_input(user_input)`
  - Validates required fields, types, ranges, and allowed values
  - Converts to proper Python types
- `create_advanced_features(user_input)`
  - Adds engineered features like `Total_Income`, `Income_to_Loan_Ratio`, etc.
- `load_model(input_size, output_dim)`
  - Instantiates the neural network and loads weights from `model.pth`

## Model

- `AdvancedLoanPredictor` (PyTorch)
  - BatchNorm → stacked Linear + BN + ReLU/GELU + Dropout layers
  - Multiple prediction heads + final prediction layer
  - Inference returns classification logits

## Training Script (train_ensemble.py)

- Loads `LAP.csv`, cleans data, creates rich engineered features
- Label-encodes categorical columns, scales features (RobustScaler)
- Trains RandomForest, GradientBoosting, LogisticRegression
- Creates a VotingClassifier ensemble and evaluates
- Saves: `ensemble_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`

## Templates

- `templates/index.html` – Input form with all required fields
- `templates/results.html` – Displays prediction, confidence, and inputs

## Artifacts

- `model.pth` – PyTorch model for inference in the Flask app
- `scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl` – preprocessing objects


