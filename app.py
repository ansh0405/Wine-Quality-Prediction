import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('wine_quality_model.pkl')

def predict_wine_quality(input_data):
    # Reshape the input data as we are predicting the label for only one instance
    input_data_reshaped = np.asarray(input_data).reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Define the app
def app():
    # Set the page title
    st.title('Wine Quality Predictor')
    # st.set_page_config(page_title="Wine Quality Prediction App", page_icon=":wine_glass:", layout="wide", initial_sidebar_state="expanded")


    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.ctfassets.net/8x8155mjsjdj/1af9dvSFEPGCzaKvs8XQ5O/a7d4adc8f9573183394ef2853afeb0b6/Copy_of_Red_Wine_Blog_Post_Header.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


    # Define the input fields
    fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=15.0, step=0.1, value=7.9)
    volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, step=0.01, value=0.32)
    citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, step=0.01, value=0.51)
    residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, step=0.1, value=1.8)
    chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, step=0.001, value=0.341)
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0, max_value=100, step=1, value=17)
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0, max_value=500, step=1, value=56)
    density = st.number_input('Density', min_value=0.9, max_value=1.2, step=0.0001, value=0.9969)
    pH = st.number_input('pH', min_value=2.0, max_value=5.0, step=0.01, value=3.04)
    sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, step=0.01, value=1.08)
    alcohol = st.number_input('Alcohol', min_value=8.0, max_value=15.0, step=0.1, value=9.2)

    # Define the feature names
    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    

    # Make a prediction
    input_data = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
    
    input_data = pd.DataFrame([input_data], columns=feature_names)
    prediction = predict_wine_quality(input_data)

    if (prediction == 1):
        st.write('<p style="text-align:center">Good Quality Wine</p>', unsafe_allow_html=True)
    else:
        st.write('<p style="text-align:center">Bad Quality Wine</p>', unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()

