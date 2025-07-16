import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Load the trained model
model = joblib.load("cardamom_date_model.pkl")

st.set_page_config(page_title="Cardamom Price Predictor", page_icon="🌿")
st.title("🌿 Cardamom Price Predictor")

st.markdown("Use this tool to predict the **average cardamom price per Kg** based on the auction date.")
st.markdown("This version allows both **single date** and **full month** predictions at once.")

# Select date for single prediction
auction_date = st.date_input("📅 Select Auction Date", value=datetime.date.today())
year = auction_date.year
month = auction_date.month
week = auction_date.isocalendar()[1]
day_of_week = auction_date.weekday()

single_input = pd.DataFrame([[year, month, week, day_of_week]], columns=['Year', 'Month', 'Week', 'DayOfWeek'])
single_prediction = model.predict(single_input)[0]

# Generate predictions for the full month of the selected date
month_dates = pd.date_range(start=auction_date.replace(day=1),
                            end=(auction_date.replace(day=1) + pd.offsets.MonthEnd(0)))

month_features = pd.DataFrame({
    'Date': month_dates,
    'Year': month_dates.year,
    'Month': month_dates.month,
    'Week': month_dates.isocalendar().week,
    'DayOfWeek': month_dates.dayofweek
})

X_month = month_features[['Year', 'Month', 'Week', 'DayOfWeek']]
y_month_pred = model.predict(X_month)
month_features['Predicted Price (Rs./Kg)'] = y_month_pred

# Display results
st.subheader("📌 Prediction for Selected Date")
st.write(f"🗓️ Date: {auction_date.strftime('%A, %d %B %Y')}")
st.success(f"💰 Predicted Price: ₹ {single_prediction:,.2f} per Kg")

st.subheader("📅 Full Month Forecast")
st.dataframe(month_features[['Date', 'Predicted Price (Rs./Kg)']].set_index('Date').style.format("{:.2f}"))
st.line_chart(month_features.set_index('Date')['Predicted Price (Rs./Kg)'])
