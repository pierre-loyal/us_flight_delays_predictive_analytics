import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import base64

# Page configuration
st.set_page_config(page_title='Flight Delay Prediction', page_icon='✈️', layout='centered')

# Load the logo image and convert it to base64
logo_path = "/Users/pierreloyal/Desktop/Bootcamp CA/us_flight_2023/icons8-airplane-take-off-100.png"
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode()

# Apply CSS for custom styling
st.markdown(f"""
    <style>
        body {{
            background-color: #C8B895; /* Specific beige background color */
            margin: 0;
            padding: 0;
        }}
        .main {{
            background-color: #C8B895;
        }}
        .center-logo {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px
        }}
        .result-text {{
            font-size: 20px;
            line-height: 1.6;
        }}
        .bold-result {{
            font-size: 24px;
            font-weight: bold;
        }}
        .center-title {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
    </style>
""", unsafe_allow_html=True)

# Display the logo at the top of the page
st.markdown(f'<div class="center-logo"><img src="data:image/png;base64,{encoded_logo}" alt="Aircraft Logo"></div>', unsafe_allow_html=True)

# Center the title
st.markdown('<div class="center-title"><h1>Flight Delay Prediction</h1></div>', unsafe_allow_html=True)

# Load the model, encoders, and accuracy
with open('flight_delay_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

with open('model_accuracy.pkl', 'rb') as accuracy_file:
    avg_accuracy = pickle.load(accuracy_file)

# Load the airport code CSV file
airport_df = pd.read_csv('/Users/pierreloyal/Desktop/Bootcamp CA/us_flight_2023/airports_geolocation.csv')

# Define function to get latitude, longitude, and city name from airport code
def get_lat_lon_city_from_airport_code(iata_code):
    result = airport_df.loc[airport_df['IATA_CODE'] == iata_code, ['LATITUDE', 'LONGITUDE', 'CITY']]
    if not result.empty:
        return result.iloc[0]['LATITUDE'], result.iloc[0]['LONGITUDE'], result.iloc[0]['CITY']
    else:
        st.error(f"Airport code '{iata_code}' not found in the database.")
        return None, None, None

# Function to get current weather conditions
def get_weather(lat, lon, api_key):
    weather_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,daily,alerts&appid={api_key}&units=metric"
    response = requests.get(weather_url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = round(weather_data['current']['temp'])
        weather_description = weather_data['current']['weather'][0]['description']
        weather_icon = weather_data['current']['weather'][0]['icon']
        icon_url = f"http://openweathermap.org/img/wn/{weather_icon}.png"
        return temperature, weather_description, icon_url
    else:
        st.error("Could not retrieve weather data. Please check the airport code and try again.")
        return None, None, None

# OpenWeatherMap API key
api_key = "c8606b2d3f511c342854d0278819b42e"

# Functions to extract features from the date/time
def get_day_of_week(date):
    return date.weekday()  # Returns 0 (Monday) to 6 (Sunday)

def get_month(date):
    return date.month - 1  # Returns 0 (January) to 11 (December)

def get_dep_time_label(time):
    hour = time.hour
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Day and month mappings for display
days_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
months = {0: 'January', 1: 'February', 2: 'March', 3: 'April', 4: 'May', 5: 'June', 6: 'July', 7: 'August', 8: 'September', 9: 'October', 10: 'November', 11: 'December'}

# Define the correct string classes based on the training data
dep_time_label_classes = ['Morning', 'Afternoon', 'Evening', 'Night']
airline_classes = ['Endeavor Air', 'American Airlines Inc.', 'Alaska Airlines Inc.', 'JetBlue Airways', 
                   'Delta Air Lines Inc', 'Frontier Airlines Inc.', 'Allegiant Air', 'Hawaiian Airlines Inc.',
                   'American Eagle Airlines Inc.', 'Spirit Air Lines', 'Southwest Airlines Co.', 'Republic Airways',
                   'PSA Airlines', 'Skywest Airlines Inc.', 'United Air Lines Inc.']
distance_type_classes = ['Short Haul >1500Mi', 'Medium Haul <3000Mi', 'Long Haul <6000Mi']

# Streamlit app
#st.title('Flight Delay Prediction')

# Input date and time
date = st.date_input('Flight Date')
time = st.time_input('Flight Time')

# Convert to datetime
date_time = datetime.combine(date, time)

# Extract features
day_of_week = get_day_of_week(date_time)
month = get_month(date_time)
dep_time_label = get_dep_time_label(time)

st.write(f'Day of the Week: {days_of_week[day_of_week]}, Month: {months[month]}, Departure Time Label: {dep_time_label}')

# User inputs for other features
airline = st.selectbox('Airline', airline_classes)
dep_airport = st.text_input('Departure Airport (3-letter IATA code)')
arr_airport = st.text_input('Arrival Airport (3-letter IATA code)')
distance_type = st.selectbox('Distance Type', distance_type_classes)

# Encode the categorical variables using the same encoding as in the model
def encode_feature(feature, value):
    if value not in encoders[feature].classes_:
        st.error(f"Value '{value}' not found in encoder classes for feature '{feature}'")
        return None
    return encoders[feature].transform([value])[0]

# Prediction button
if st.button('Predict Delay Type'):
    dep_time_label_encoded = encode_feature('DepTime_label', dep_time_label)
    airline_encoded = encode_feature('Airline', airline)
    dep_airport_encoded = encode_feature('Dep_Airport', dep_airport)
    arr_airport_encoded = encode_feature('Arr_Airport', arr_airport)
    distance_type_encoded = encode_feature('Distance_type', distance_type)

    if None not in (dep_time_label_encoded, airline_encoded, dep_airport_encoded, arr_airport_encoded, distance_type_encoded):
        # Construct feature array
        features = np.array([day_of_week, month, dep_time_label_encoded, airline_encoded, dep_airport_encoded, arr_airport_encoded, distance_type_encoded]).reshape(1, -1)
        prediction = model.predict(features)
        
        # Get latitude, longitude, and city name from arrival airport code
        lat, lon, city_name = get_lat_lon_city_from_airport_code(arr_airport)
        if lat and lon:
            # Get current weather conditions for the arrival city
            temperature, weather_description, icon_url = get_weather(lat, lon, api_key)
            if temperature and weather_description and icon_url:
                st.markdown(f'''
                    <div class="result-text">
                        Predicted Departure Delay Type: <span class="bold-result">{prediction[0]}</span><br><br>
                        Model Accuracy: <span class="bold-result">{int(avg_accuracy*100)}%</span><br><br>
                        Current Weather in {city_name}: {weather_description}, {temperature}°C
                    </div>
                ''', unsafe_allow_html=True)
                st.image(icon_url, caption=f"{weather_description.capitalize()}, {temperature}°C")
    else:
        st.error("Some inputs could not be encoded. Please check your inputs and try again.")
