import torch
import torch.nn as nn
import joblib
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load target label encoder (if classification)
try:
    le_target = joblib.load('label_encoder_target.pkl')
    is_classification = True
except:
    le_target = None
    is_classification = False

# Sample input dictionary (change this with real user input or input form)
# user_input = {
#     "Loan_ID": "LP001002",
#     "Gender": "Male",
#     "Married": "Yes",
#     "Dependents": 0,
#     "Education": "Graduate",
#     "Self_Employed": "No",
#     "ApplicantIncome": 3000,
#     "CoapplicantIncome": 0,
#     "LoanAmount": 12000,
#     "Loan_Amount_Term": 370,
#     "Credit_History": 1,
#     "Property_Area": "Urban",
# }

user_input = {}

user_input["Loan_ID"] = input("Loan ID: ")
user_input["Gender"] = input("Gender (Male/Female): ")
user_input["Married"] = input("Married (Yes/No): ")
user_input["Dependents"] = int(input("Number of Dependents: "))
user_input["Education"] = input("Education (Graduate/Not Graduate): ")
user_input["Self_Employed"] = input("Self Employed (Yes/No): ")
user_input["ApplicantIncome"] = float(input("Applicant Income: "))
user_input["CoapplicantIncome"] = float(input("Coapplicant Income: "))
user_input["LoanAmount"] = float(input("Loan Amount: "))
user_input["Loan_Amount_Term"] = float(input("Loan Amount Term (in days): "))
user_input["Credit_History"] = int(input("Credit History (1 = Good, 0 = Bad): "))
user_input["Property_Area"] = input("Property Area (Urban/Semiurban/Rural): ")

print("\nCollected Input:")
print(user_input)

# 3. Convert user input into a DataFrame row
import pandas as pd
input_df = pd.DataFrame([user_input])
print(input_df)
# 4. Encode categorical fields
# Encode categorical fields only if present in input
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            print(f"[ERROR] Unknown label in column '{col}': {input_df[col].values[0]}")
            exit()
    elif input_df[col].dtype == 'object':
        print(f"[ERROR] Missing encoder for column '{col}' — did you train with it?")
        exit()



# 5. Scale numerical data
X_input = scaler.transform(input_df.values.astype(np.float32))
X_input = torch.tensor(X_input, dtype=torch.float32)

# 6. Load model
input_size = X_input.shape[1]
print(f"Input size: {input_size}")
output_dim = len(le_target.classes_) if is_classification else 1
model = SimpleNN(input_dim=input_size, output_dim=output_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 7. Predict
with torch.no_grad():
    output = model(X_input)
    if is_classification:
        pred = output.argmax(dim=1).item()
        prediction = le_target.inverse_transform([pred])[0]
    else:
        prediction = output.item()

print(f"Predicted Output: {prediction}")
