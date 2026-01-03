from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/churn_scaler.pkl')

# EXACT Columns used in training
model_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 
    'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
    'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # 1. Initialize a dictionary with all columns set to 0
        input_data = {col: [0] for col in model_columns}

        # 2. Extract Numeric Features
        input_data['tenure'] = [float(request.form['tenure'])]
        input_data['MonthlyCharges'] = [float(request.form['MonthlyCharges'])]
        input_data['TotalCharges'] = [float(request.form['TotalCharges'])]
        input_data['SeniorCitizen'] = [int(request.form['SeniorCitizen'])]

        # 3. Extract & Encode Categorical Features
        # (We set the specific column to 1 based on user choice)

        if request.form['gender'] == 'Male': input_data['gender_Male'] = [1]
        if request.form['Partner'] == 'Yes': input_data['Partner_Yes'] = [1]
        if request.form['Dependents'] == 'Yes': input_data['Dependents_Yes'] = [1]
        if request.form['PhoneService'] == 'Yes': input_data['PhoneService_Yes'] = [1]
        if request.form['PaperlessBilling'] == 'Yes': input_data['PaperlessBilling_Yes'] = [1]

        # Multiple Lines
        lines = request.form['MultipleLines']
        if lines == 'Yes': input_data['MultipleLines_Yes'] = [1]
        elif lines == 'No phone service': input_data['MultipleLines_No phone service'] = [1]

        # Internet Service (DSL is default 0, 0)
        internet = request.form['InternetService']
        if internet == 'Fiber optic': input_data['InternetService_Fiber optic'] = [1]
        elif internet == 'No': input_data['InternetService_No'] = [1]

        # Helper function for the 6 services that have "No internet service" option
        def set_service(val, prefix):
            if val == 'Yes': input_data[f'{prefix}_Yes'] = [1]
            elif val == 'No internet service': input_data[f'{prefix}_No internet service'] = [1]

        set_service(request.form['OnlineSecurity'], 'OnlineSecurity')
        set_service(request.form['OnlineBackup'], 'OnlineBackup')
        set_service(request.form['DeviceProtection'], 'DeviceProtection')
        set_service(request.form['TechSupport'], 'TechSupport')
        set_service(request.form['StreamingTV'], 'StreamingTV')
        set_service(request.form['StreamingMovies'], 'StreamingMovies')

        # Contract (Month-to-month is default 0, 0)
        contract = request.form['Contract']
        if contract == 'One year': input_data['Contract_One year'] = [1]
        elif contract == 'Two year': input_data['Contract_Two year'] = [1]

        # Payment Method (Bank transfer is default 0, 0, 0)
        payment = request.form['PaymentMethod']
        if payment == 'Credit card (automatic)': input_data['PaymentMethod_Credit card (automatic)'] = [1]
        elif payment == 'Electronic check': input_data['PaymentMethod_Electronic check'] = [1]
        elif payment == 'Mailed check': input_data['PaymentMethod_Mailed check'] = [1]

        # 4. Create DataFrame
        final_df = pd.DataFrame(input_data)

        # 5. Scale Logic
        # (StandardScaler expects column names to match)
        new_data_scaled = scaler.transform(final_df)

        # 6. Predict
        prediction = model.predict(new_data_scaled)
        
        # Get Probability (Confidence) e.g., "85% Risk"
        probability = model.predict_proba(new_data_scaled)[0][1] * 100

        result_text = "Churn (High Risk 🚨)" if prediction[0] == 1 else "No Churn (Safe ✅)"

        return render_template("home.html", prediction_text=f"{result_text}", probability=f"{probability:.2f}% Chance of Leaving")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)