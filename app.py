import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('house_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')
defaults = joblib.load('default_values.pkl')

st.title("🏠 House Price Predictor")
st.write("Ghar ki details daalo, price predict ho jaayegi.")

overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", min_value=0, value=1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=800)
first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=0, value=1000)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=8000)
total_rooms = st.slider("Total Rooms Above Ground", 2, 15, 6)
fireplaces = st.slider("Fireplaces", 0, 4, 1)

if st.button("Predict Price"):
    input_row = defaults.copy()

    input_row['OverallQual'] = overall_qual
    input_row['GrLivArea'] = gr_liv_area
    input_row['GarageCars'] = garage_cars
    input_row['TotalBsmtSF'] = total_bsmt_sf
    input_row['1stFlrSF'] = first_flr_sf
    input_row['FullBath'] = full_bath
    input_row['YearBuilt'] = year_built
    input_row['LotArea'] = lot_area
    input_row['TotRmsAbvGrd'] = total_rooms
    input_row['Fireplaces'] = fireplaces

    input_df = pd.DataFrame([input_row])[model_columns]

    prediction_log = model.predict(input_df)
    prediction_price = np.expm1(prediction_log)

    st.success(f"### Predicted Price: ${prediction_price[0]:,.2f}")
