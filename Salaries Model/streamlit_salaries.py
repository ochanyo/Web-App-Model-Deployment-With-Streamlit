import streamlit as st
import numpy as np
import joblib

st.write("SALARIES PREDICTION")

exp=st.number_input('Input Your Years Of Experience Here:',0.00,step=1.00)
exp=np.array(exp).reshape(-1,1)


model=joblib.load('salaries_model.pkl')
pred=model.predict(exp)[0]

if st.button('Predict'):
    st.success(f"Your Predicted Salary is {round(pred)}")