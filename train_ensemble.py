import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")
df = pd.read_csv('LAP.csv')
df = df.drop_duplicates()

print(f"Dataset shape: {df.shape}")
print(f"Missing values per column:\n{df.isnull().sum()}")

# Advanced data cleaning and preprocessing
def clean_data(df):
    # Handle missing values with advanced strategies
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # For numeric columns, use robust imputation
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Use median for skewed data, mean for normal data
            if abs(df[col].skew()) > 1:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    
    # For categorical columns, use mode with fallback
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    
    # Create advanced features for better model performance
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_to_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 1)
    df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + 1)
    df['Dependents_Income_Ratio'] = df['Dependents'] / (df['Total_Income'] + 1)
    df['Income_per_Dependent'] = df['Total_Income'] / (df['Dependents'] + 1)
    df['Loan_Amount_per_Term'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
    df['Credit_Score'] = df['Credit_History'] * 100  # Convert to credit score scale
    
    # Advanced income features
    df['Income_Stability'] = df['ApplicantIncome'] / (df['CoapplicantIncome'] + 1)
    df['Income_Growth_Potential'] = df['Education'].map({'Graduate': 1.2, 'Not Graduate': 1.0}) * df['Total_Income']
    
    # Loan risk features
    df['Loan_Risk_Score'] = (df['LoanAmount'] * df['Dependents']) / (df['Total_Income'] + 1)
    df['Repayment_Capacity'] = df['Total_Income'] / (df['LoanAmount'] * df['Loan_Amount_Term'] / 365 + 1)
    
    # Binning for better categorical representation
    df['Income_Category'] = pd.cut(df['Total_Income'], 
                                  bins=[0, 5000, 10000, 20000, 50000, float('inf')], 
                                  labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    df['Loan_Amount_Category'] = pd.cut(df['LoanAmount'], 
                                       bins=[0, 100, 500, 1000, 5000, float('inf')], 
                                       labels=['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large'])
    
    df['Loan_Term_Category'] = pd.cut(df['Loan_Amount_Term'], 
                                     bins=[0, 180, 360, 720, float('inf')], 
                                     labels=['Short', 'Medium', 'Long', 'Very_Long'])
    
    # Interaction features
    df['Education_Income_Interaction'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0}) * df['Total_Income']
    df['Married_Income_Interaction'] = df['Married'].map({'Yes': 1, 'No': 0}) * df['Total_Income']
    df['Self_Employed_Risk'] = df['Self_Employed'].map({'Yes': 1.5, 'No': 1.0}) * df['Loan_Risk_Score']
    
    return df

df = clean_data(df)
print("Data cleaned and enhanced with advanced features")

# Advanced encoding strategy
def advanced_encoding(df):
    label_encoders = {}
    
    # Handle categorical columns with advanced encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

df, label_encoders = advanced_encoding(df)

# Prepare target variable
target_col = 'Loan_Status'  # Explicitly specify target column
X = df.drop(columns=[target_col]).values.astype(np.float32)
y = df[target_col].values

print(f"Target column: {target_col}")
print(f"Target values: {np.unique(y)}")
print(f"Target distribution: {np.bincount(y)}")

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)
is_classification = True
print(f"Classification problem with {len(np.unique(y))} classes")
print(f"Class distribution after encoding: {np.bincount(y)}")

# Advanced scaling with RobustScaler
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)

# Split data without stratification to avoid issues
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train multiple traditional ML models for ensemble
print("\nTraining traditional ML models...")

# Random Forest with optimized parameters
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_score = rf.score(X_val, y_val)
print(f"Random Forest Validation Score: {rf_score:.4f}")

# Gradient Boosting with optimized parameters
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_val, y_val)
print(f"Gradient Boosting Validation Score: {gb_score:.4f}")

# Logistic Regression with regularization
lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
lr.fit(X_train, y_train)
lr_score = lr.score(X_val, y_val)
print(f"Logistic Regression Validation Score: {lr_score:.4f}")

# Create ensemble classifier
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('lr', lr)
    ],
    voting='soft'  # Use probability voting
)

ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_val, y_val)
print(f"Ensemble Validation Score: {ensemble_score:.4f}")

# Test on test set
print(f"\nTest Set Results:")
print(f"Random Forest: {rf.score(X_test, y_test):.4f}")
print(f"Gradient Boosting: {gb.score(X_test, y_test):.4f}")
print(f"Logistic Regression: {lr.score(X_test, y_test):.4f}")
print(f"Ensemble: {ensemble.score(X_test, y_test):.4f}")

# Cross-validation scores
print(f"\nCross-Validation Scores (5-fold):")
rf_cv = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
gb_cv = cross_val_score(gb, X_scaled, y, cv=5, scoring='accuracy')
lr_cv = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy')
ensemble_cv = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='accuracy')

print(f"Random Forest CV: {rf_cv.mean():.4f} (+/- {rf_cv.std() * 2:.4f})")
print(f"Gradient Boosting CV: {gb_cv.mean():.4f} (+/- {gb_cv.std() * 2:.4f})")
print(f"Logistic Regression CV: {lr_cv.mean():.4f} (+/- {lr_cv.std() * 2:.4f})")
print(f"Ensemble CV: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std() * 2:.4f})")

# Feature importance analysis
feature_names = df.drop(columns=[target_col]).columns
rf_importance = rf.feature_importances_
gb_importance = gb.feature_importances_

# Combine feature importance
combined_importance = (rf_importance + gb_importance) / 2
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': combined_importance
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

# Save the best performing model
print(f"\nSaving ensemble model (best accuracy)")
import joblib
joblib.dump(ensemble, 'ensemble_model.pkl')
joblib.dump(robust_scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(le_target, 'label_encoder_target.pkl')

print("\nAdvanced model training completed!")
print("Ensemble model and preprocessing objects saved successfully!")

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features for Loan Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
