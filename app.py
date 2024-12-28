import streamlit as st
import pandas as pd
from joblib import load  # Import load function from joblib to load the model
from sklearn.preprocessing import StandardScaler

# Function to predict Close price
def predict_close_price(open_price, high_price, low_price, adj_close, volume, model, scaler):
    # Scale the input values
    scaled_input = scaler.fit_transform([[open_price, high_price, low_price, adj_close, volume]])
    
    # Use the scaled input to predict Close price
    predicted_close = model.predict(scaled_input)
    
    return predicted_close[0]  # Return the predicted Close price

def main():
    st.title('Close Price Prediction')

    # Load the trained model and scaler
    model_file = 'lr_model.pkl'  
    model = load(model_file)

    scaler_file = 'scaler_standard.joblib'  # Replace with your scaler file
    scaler = load(scaler_file)

    # User inputs for Open, High, Low, Adj Close, and Volume
    open_price = st.number_input('Enter Open Price', min_value=19028.36, format="%.2f")
    high_price = st.number_input('Enter High Price', min_value=19121.01, format="%.2f")
    low_price = st.number_input('Enter Low Price', min_value=18213.65, format="%.2f")
    adj_close = st.number_input('Enter Adjusted Close Price', min_value=18591.93, format="%.2f")
    volume = st.number_input('Enter Volume', min_value=86150000.00, format="%.2f")

    # Predict Close price when the Predict button is clicked
    if st.button('Predict'):
        # Scale the input values using the loaded scaler
        predicted_close = predict_close_price(open_price, high_price, low_price, adj_close, volume, model, scaler)
        st.success(f'Predicted Close Price: {predicted_close:.2f}')

if __name__ == '__main__':
    main()
