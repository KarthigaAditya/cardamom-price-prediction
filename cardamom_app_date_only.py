import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Load the trained model (make sure it's in the same folder)
model = joblib.load("cardamom_date_model.pkl")

# Streamlit app settings
st.set_page_config(page_title="Cardamom Price Predictor ", page_icon="🌿")
st.title("🌿 Cardamom Price Predictor South Indian Green Cardamom Company Ltd, Kochi	(Date-Based)")

st.markdown(
    "Predict the **expected average cardamom price per Kg** using just the auction date. "
    "This tool is designed for farmers and traders who may not know exact auction quantities or lots."
)

# User inputs: just the auction date
auction_date = st.date_input("📅 Select Auction Date", value=datetime.date.today())

# Extract features from date
year = auction_date.year
month = auction_date.month
week = auction_date.isocalendar()[1]
day_of_week = auction_date.weekday()  # 0=Monday, 6=Sunday

# Prepare input for the model
input_data = pd.DataFrame([[year, month, week, day_of_week]],
                          columns=['Year', 'Month', 'Week', 'DayOfWeek'])

# Predict
if st.button("Predict Cardamom Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Predicted Avg. Cardamom Price: ₹{predicted_price:,.2f} per Kg")
