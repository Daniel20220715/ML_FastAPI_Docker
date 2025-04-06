import streamlit as st
import joblib 

# Load the model and scaler
model = joblib.load ('linear_regression_model.pkl')
scaler = joblib.load ('scaler.pkl')

#Streamlit app
st.title("MetaBrains Student Test Score Prediction")
st.write("Enter the number of hours studied to predict the test score.")

#User input
hours = st.number_input("Hours Studied", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        data = pd.DataFrame([[hours]], columns=['Hours_Studied'])
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.write("Predicted Test Score: ", prediction[0])
    except Exception as e:
        st.write("An error occurred:", e)
