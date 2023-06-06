from flask import Flask, render_template, request
import pickle
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Load the trained model
with open('best_GB.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the form
    gender = float(request.form['gender'])
    married = request.form['married']
    dependents = request.form['dependents']
    education = request.form['education']
    self_employed = request.form['self_employed']
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = float(request.form['loan_amount_term'])
    credit_history = float(request.form['credit_history'])
    property_area = request.form['property_area']

    # Create a dictionary of input values
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Perform any necessary data preprocessing on the inputs
    # (e.g., converting categorical variables to numerical)

    # Make the prediction using the loaded model
    prediction = model.predict([list(input_data.values())])[0]

    # Convert the prediction to a human-readable output
    if prediction == 'Y':
        result = 'Approved'
    else:
        result = 'Not Approved'

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
