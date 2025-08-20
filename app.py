from flask import Flask, render_template, request, redirect, url_for, session
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Load preprocessing objects
robust_scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

try:
    le_target = joblib.load('label_encoder_target.pkl')
    is_classification = True
except:
    le_target = None
    is_classification = False

# Advanced Neural Network Architecture (same as training script)
class AdvancedLoanPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], output_dim=2, dropout_rate=0.3):
        super(AdvancedLoanPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Build dynamic layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function (mix of ReLU and GELU)
            if i % 2 == 0:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
            
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multiple prediction heads for ensemble effect
        self.prediction_heads = nn.ModuleList([
            nn.Linear(prev_dim, output_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Linear(prev_dim, output_dim)
        ])
        
        # Final prediction layer
        self.final_prediction = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_dim // 2, output_dim)
        )
    
    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Multiple prediction heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(features)
            predictions.append(pred)
        
        # Ensemble predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        # Final prediction
        final_pred = self.final_prediction(features)
        
        # Return ensemble of all predictions
        return final_pred

# Load the model
def load_model(input_size, output_dim):
    model = AdvancedLoanPredictor(input_dim=input_size, output_dim=output_dim)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Advanced input validation and preprocessing
def validate_and_preprocess_input(user_input):
    """Advanced input validation and preprocessing"""
    processed_input = {}
    
    # Define expected ranges and validation rules
    validation_rules = {
        'Loan_ID': {'type': str, 'min_length': 1, 'max_length': 50},
        'Gender': {'type': str, 'allowed_values': ['Male', 'Female']},
        'Married': {'type': str, 'allowed_values': ['Yes', 'No']},
        'Dependents': {'type': int, 'min': 0, 'max': 10},
        'Education': {'type': str, 'allowed_values': ['Graduate', 'Not Graduate']},
        'Self_Employed': {'type': str, 'allowed_values': ['Yes', 'No']},
        'ApplicantIncome': {'type': float, 'min': 0, 'max': 1000000},
        'CoapplicantIncome': {'type': float, 'min': 0, 'max': 1000000},
        'LoanAmount': {'type': float, 'min': 0, 'max': 1000000},
        'Loan_Amount_Term': {'type': float, 'min': 0, 'max': 10000},
        'Credit_History': {'type': int, 'allowed_values': [0, 1]},
        'Property_Area': {'type': str, 'allowed_values': ['Urban', 'Semiurban', 'Rural']}
    }
    
    for field, rules in validation_rules.items():
        value = user_input.get(field)
        
        if value is None or (isinstance(value, str) and value.strip() == ''):
            raise ValueError(f"Field '{field}' is required")
        
        # Type conversion and validation
        try:
            if rules['type'] == int:
                processed_input[field] = int(value)
            elif rules['type'] == float:
                processed_input[field] = float(value)
            else:
                processed_input[field] = str(value).strip()
        except (ValueError, TypeError):
            raise ValueError(f"Field '{field}' must be of type {rules['type'].__name__}")
        
        # Range validation
        if 'min' in rules and processed_input[field] < rules['min']:
            raise ValueError(f"Field '{field}' must be at least {rules['min']}")
        if 'max' in rules and processed_input[field] > rules['max']:
            raise ValueError(f"Field '{field}' must be at most {rules['max']}")
        
        # Length validation for strings
        if 'min_length' in rules and len(processed_input[field]) < rules['min_length']:
            raise ValueError(f"Field '{field}' must be at least {rules['min_length']} characters")
        if 'max_length' in rules and len(processed_input[field]) > rules['max_length']:
            raise ValueError(f"Field '{field}' must be at most {rules['max_length']} characters")
        
        # Allowed values validation
        if 'allowed_values' in rules and processed_input[field] not in rules['allowed_values']:
            raise ValueError(f"Field '{field}' must be one of: {', '.join(map(str, rules['allowed_values']))}")
    
    return processed_input

# Advanced feature engineering
def create_advanced_features(user_input):
    """Create advanced features for better prediction"""
    features = user_input.copy()
    
    # Create derived features
    features['Total_Income'] = features['ApplicantIncome'] + features['CoapplicantIncome']
    features['Income_to_Loan_Ratio'] = features['Total_Income'] / (features['LoanAmount'] + 1)
    features['Loan_to_Income_Ratio'] = features['LoanAmount'] / (features['Total_Income'] + 1)
    features['Dependents_Income_Ratio'] = features['Dependents'] / (features['Total_Income'] + 1)
    
    # Create categorical features
    if features['Total_Income'] <= 5000:
        features['Income_Category'] = 'Very_Low'
    elif features['Total_Income'] <= 10000:
        features['Income_Category'] = 'Low'
    elif features['Total_Income'] <= 20000:
        features['Income_Category'] = 'Medium'
    elif features['Total_Income'] <= 50000:
        features['Income_Category'] = 'High'
    else:
        features['Income_Category'] = 'Very_High'
    
    if features['LoanAmount'] <= 100:
        features['Loan_Amount_Category'] = 'Very_Small'
    elif features['LoanAmount'] <= 500:
        features['Loan_Amount_Category'] = 'Small'
    elif features['LoanAmount'] <= 1000:
        features['Loan_Amount_Category'] = 'Medium'
    elif features['LoanAmount'] <= 5000:
        features['Loan_Amount_Category'] = 'Large'
    else:
        features['Loan_Amount_Category'] = 'Very_Large'
    
    return features

# Predefined input fields (names and types)
INPUT_FIELDS = {
    "Loan_ID": str,
    "Gender": str,
    "Married": str,
    "Dependents": int,
    "Education": str,
    "Self_Employed": str,
    "ApplicantIncome": float,
    "CoapplicantIncome": float,
    "LoanAmount": float,
    "Loan_Amount_Term": float,
    "Credit_History": int,
    "Property_Area": str
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Collect and validate inputs
            user_input = {}
            for field, ftype in INPUT_FIELDS.items():
                value = request.form.get(field)
                if value is None or value.strip() == '':
                    return render_template('index.html', error=f"Please fill the field: {field}", inputs=request.form)
                user_input[field] = value
            
            # Advanced validation and preprocessing
            try:
                user_input = validate_and_preprocess_input(user_input)
            except ValueError as e:
                return render_template('index.html', error=str(e), inputs=request.form)
            
            # Create advanced features
            user_input = create_advanced_features(user_input)
            
            # Convert to DataFrame for processing
            input_df = pd.DataFrame([user_input])
            
            # Encode categoricals with fallback handling
            for col in input_df.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    try:
                        # Handle unseen categories gracefully
                        if col in ['Income_Category', 'Loan_Amount_Category']:
                            # For new categories, assign a default encoding
                            if input_df[col].values[0] not in le.classes_:
                                input_df[col] = le.transform(['Medium'])[0]  # Default to medium
                            else:
                                input_df[col] = le.transform(input_df[col])
                        else:
                            input_df[col] = le.transform(input_df[col])
                    except ValueError:
                        # Handle unseen categories by adding them to the encoder
                        all_categories = list(le.classes_) + [input_df[col].values[0]]
                        le.fit(all_categories)
                        input_df[col] = le.transform(input_df[col])
                
                elif input_df[col].dtype == 'object':
                    # Missing encoder for categorical column
                    return render_template('index.html', error=f"Missing encoder for column '{col}'.", inputs=request.form)
            
            # Scale numerical data with robust scaling
            try:
                X_input = robust_scaler.transform(input_df.values.astype(np.float32))
            except Exception as e:
                return render_template('index.html', error=f"Scaling error: {str(e)}", inputs=request.form)
            
            X_input = torch.tensor(X_input, dtype=torch.float32)
            
            # Prepare model
            input_size = X_input.shape[1]
            output_dim = len(le_target.classes_) if is_classification else 1
            model = load_model(input_size, output_dim)
            
            # Predict with confidence
            with torch.no_grad():
                output = model(X_input)
                if is_classification:
                    # Get prediction probabilities
                    probabilities = F.softmax(output, dim=1)
                    confidence, pred = torch.max(probabilities, 1)
                    prediction = le_target.inverse_transform([pred.item()])[0]
                    confidence_score = confidence.item()
                else:
                    prediction = output.item()
                    confidence_score = 0.8  # Default confidence for regression
            
            # Store results in session and redirect to results page
            session['prediction'] = str(prediction) if prediction is not None else None
            session['confidence'] = confidence_score
            
            # Convert user_input values to native Python types for session storage
            user_input_clean = {}
            for key, value in user_input.items():
                if hasattr(value, 'item'):  # numpy scalar
                    user_input_clean[key] = value.item()
                else:
                    user_input_clean[key] = value
            
            session['user_input'] = user_input_clean
            return redirect(url_for('results'))
            
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}", inputs=request.form)
    
    return render_template('index.html', inputs={})

@app.route('/results')
def results():
    prediction = session.get('prediction')
    user_input = session.get('user_input')
    confidence = session.get('confidence', 0.8)
    
    if prediction is None or user_input is None:
        return redirect(url_for('home'))
    
    # Determine eligibility based on prediction with confidence threshold
    pred_str = str(prediction).lower()
    is_eligible = pred_str in ['y', 'yes', 'approved', '1', 'true', 'approve', 'loan approved']
    
    # Adjust eligibility based on confidence
    if confidence < 0.6:  # Low confidence
        is_eligible = False  # Conservative approach for low confidence
    
    # Clear session data after displaying results
    session.pop('prediction', None)
    session.pop('user_input', None)
    session.pop('confidence', None)
    
    return render_template('results.html', 
                         prediction=prediction, 
                         user_input=user_input, 
                         is_eligible=is_eligible,
                         confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)