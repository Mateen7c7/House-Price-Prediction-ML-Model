import streamlit as st
import joblib

model = joblib.load("house_price_model.pkl")

st.title("House Price Prediction")

MedInc = st.number_input("Median Income", value=8.3252)
HouseAge = st.number_input("House Age", value=41.0)
AveRooms = st.number_input("Average Rooms", value=6.9841)
AveBedrms = st.number_input("Average Bedrooms", value=1.0238)
Population = st.number_input("Population", value=322.0)
AveOccup = st.number_input("Average Occupancy", value=2.5555)
Latitude = st.number_input("Latitude", value=37.88)
Longitude = st.number_input("Longitude", value=-122.23)

if st.button("Predict"):
    prediction = model.predict([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])

    st.success(f"Predicted House Price: ${prediction[0] * 100000:,.2f}")