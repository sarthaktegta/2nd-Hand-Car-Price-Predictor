# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:18:20 2024

@author: Sarthak Tegta
"""

import os
import pickle
import streamlit as st
import numpy as np

model_path = 'trained_model.sav'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        loaded_model = None
else:
    st.error(f"Model file not found: {model_path}")
    loaded_model = None


def car_predict(input_data):
    input_data_as_np = np.asarray(input_data, dtype=float)
    input_data_reshape = input_data_as_np.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)
    return np.exp(prediction)  # Ensure this is the correct transformation for your model

def main():
    st.title('2nd hand Car price Prediction Web APP')

    # Inputs for the model
    Mileage = st.number_input('Mileage of the Car', min_value=0)
    Engine_V = st.number_input('Volume of the Engine', min_value=0.0, format="%.1f")

    # Brand inputs (0 for False, 1 for True)
    Brand_BMW = st.selectbox('BMW?', [0, 1])
    Brand_Mercedes = st.selectbox('Mercedes?', [0, 1])
    Brand_Mistubi = st.selectbox('Mistubi?', [0, 1])
    Brand_Renault = st.selectbox('Renault?', [0, 1])
    Brand_Toyota = st.selectbox('Toyota?', [0, 1])
    Brand_Volkswagen = st.selectbox('Volkswagen?', [0, 1])

    # Body type inputs (0 for False, 1 for True)
    Body_hatch = st.selectbox('Body_hatch?', [0, 1])
    Body_other = st.selectbox('Body_other?', [0, 1])
    Body_sedan = st.selectbox('Body_sedan?', [0, 1])
    Body_vagon = st.selectbox('Body_vagon?', [0, 1])
    Body_van = st.selectbox('Body_van?', [0, 1])

    # Engine type inputs (0 for False, 1 for True)
    Engine_Type_Gas = st.selectbox('Engine_Type_Gas?', [0, 1])
    Engine_Type_Other = st.selectbox('Engine_Type_Other?', [0, 1])
    Engine_Type_Petrol = st.selectbox('Engine_Type_Petrol?', [0, 1])

    # Registration input (0 for False, 1 for True)
    Registration_yes = st.selectbox('Registration_yes?', [0, 1])

    prediction = ''
    
    # Button for prediction
    if st.button('Predict the Car price'):
        input_data = [Mileage, Engine_V, Brand_BMW, Brand_Mercedes, Brand_Mistubi,
                      Brand_Renault, Brand_Toyota, Brand_Volkswagen, Body_hatch,
                      Body_other, Body_sedan, Body_vagon, Body_van, Engine_Type_Gas,
                      Engine_Type_Other, Engine_Type_Petrol, Registration_yes]
        
        prediction = car_predict(input_data)
        
    st.success(f'The predicted car price is $: {prediction}')

if __name__ == '__main__':
    main()
