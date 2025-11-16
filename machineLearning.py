import streamlit as st
import pandas as pd
import numpy as np
import pickle  # to load a saved model
import base64  # to handle gif encoding

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict[val]

def get_value(val, my_dict):
    return my_dict[val]

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])


if app_mode == 'Home':
    st.title('Loan Prediction')
    st.image('loan.png')
    st.markdown('Dataset:')
    data = pd.read_csv('loan.csv')
    st.write(data.head())
    st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(20))

if app_mode == 'Prediction':
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', 0, 10000, 0)
    LoanAmount = st.sidebar.slider('LoanAmount in K$', 9.0, 700.0, 200.0)
    # Assuming additional input features here...
    # Prediction Logic
    if st.button("Predict"):
        loaded_model = pickle.load(open('LG.sav', 'rb'))
        prediction = loaded_model.predict(np.array([ApplicantIncome, LoanAmount]).reshape(1, -1))
        if prediction[0] == 0:
            st.error('According to our calculations, you will not get the loan.')
        else:
            st.success('Congratulations! You will get the loan.')