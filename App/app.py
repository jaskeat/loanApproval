import streamlit as st
import joblib
import numpy as np

# Load the model from the file
try:
    import os
    model_path = os.path.join(os.path.dirname(__file__), '../Model/decision_tree_model.pkl')
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'decision_tree_model.pkl' is in the correct directory.")
    st.stop()

def predict_loan_approval(person_income, loan_amnt, person_emp_exp, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score):
    # Prepare the input data
    input_data = np.array([[person_income, loan_amnt, person_emp_exp, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score]])
    
    
    # Make the prediction
    prediction = model.predict(input_data)
    print(prediction)
    return prediction[0]

st.title('Loan Approval  Prediction')

col1, col2, col3, col4 = st.columns(4)

with col1:
    person_income = st.number_input('Person Yearly Income ($USD)', format="%f", step=0.01, placeholder='Yearly income in USD')
    loan_int_rate = st.number_input('Loan Interest Rate (%)', format="%f", step=1.0)

with col2:
    person_emp_exp = st.slider('Person Employment Experience (Years)', min_value=0, max_value=75, step=1)
    credit_score = st.number_input('Credit Score', step=1, format="%d")

with col3:
    cb_person_cred_hist_length = st.slider('Credit Bureau Person Credit History Length (Years)', min_value=0, max_value=75, step=1)

with col4:
    loan_amnt = st.number_input('Loan Amount ($USD)', format="%f", step=0.01)


# Add the loan percent income by calculating from the stuff given

if person_income!= 0:
    loan_percent_income = loan_amnt/person_income
else:
    loan_percent_income = 0
    
if st.button('Submit'):
    prediction = predict_loan_approval(person_income, loan_amnt, person_emp_exp, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score)
    if prediction >= 0.5:
        st.markdown('<h1 style="color:green;">Prediction: Approved</h1>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 style="color:red;">Prediction: Rejected</h1>', unsafe_allow_html=True)
