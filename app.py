import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load the pre-trained model
model = pickle.load(open('flight_rf.pkl', 'rb'))



# Streamlit app title
st.title("Flight Price Prediction")
st.image("flight.jpg")

# Input fields
st.header("Input Parameters")

# Airline
Airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet'])

# Date of Journey
Date_of_Journey = st.date_input("Date of Journey")

# Extract date, month, and year from Date_of_Journey
Date = Date_of_Journey.day
Month = Date_of_Journey.month
Year = Date_of_Journey.year

# Source
Source = st.selectbox("Source", ["Bangalore", "Kolkata", "Delhi", "Chennai", "Mumbai"])

# Destination
Destination = st.selectbox("Destination", ["New Delhi", "Bangalore", "Cochin", "Kolkata", "Delhi", "Hyderabad"])

# Total Stops
Total_Stops = st.selectbox("Total Stops", ["non-stop", "2 stops", "1 stop", "3 stops", "4 stops"])

# Additional Info
Additional_Info = st.selectbox("Additional Info", [
    "No info", "In-flight meal not included", "No check-in baggage included",
    "1 Short layover", "No Info", "1 Long layover", "Change airports",
    "Business class", "Red-eye flight", "2 Long layover"
])

# Arrival Time (You can customize this based on your DataFrame columns)
Arrival_hour = st.number_input("Arrival Hour")
Arrival_min = st.number_input("Arrival Minute")

# Departure Time (You can customize this based on your DataFrame columns)
Dept_hour = st.number_input("Departure Hour")
Dept_min = st.number_input("Departure Minute")

# Duration (You can customize this based on your DataFrame columns)
Duration_hour = st.number_input("Duration Hour")
Duration_min = st.number_input("Duration Minute")

# Create a dictionary with user inputs
input_data = {
    'Airline': Airline,
    'Date': Date,
    'Month': Month,
    'Year': Year,
    'Source': Source,
    'Destination': Destination,
    'Total_Stops': Total_Stops,
    'Additional_Info': Additional_Info,
    'Arrival_hour': Arrival_hour,
    'Arrival_min': Arrival_min,
    'Dept_hour': Dept_hour,
    'Dept_min': Dept_min,
    'Duration_hour': Duration_hour,
    'Duration_min': Duration_min
}

# Function to preprocess input data
def preprocess_data(df):
    # Check if 'Date_of_Journey' column exists in the DataFrame
    if 'Date_of_Journey' in df.columns:
        df['Date'] = df['Date_of_Journey'].str.split('/').str[0]
        df['Month'] = df['Date_of_Journey'].str.split('/').str[1]
        df['Year'] = df['Date_of_Journey'].str.split('/').str[2]
        df.drop('Date_of_Journey', axis=1, inplace=True)
    # Check if 'Arrival_Time' column exists in the DataFrame
    if 'Arrival_Time' in df.columns:
        df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
        df['Arrival_hour'] = df['Arrival_Time'].str.split(':').str[0]
        df['Arrival_min'] = df['Arrival_Time'].str.split(':').str[1]
    # Check if 'Dep_Time' column exists in the DataFrame
    if 'Dep_Time' in df.columns:
        df['Dept_hour'] = df['Dep_Time'].str.split(':').str[0]
        df['Dept_min'] = df['Dep_Time'].str.split(':').str[1]
        df.drop('Dep_Time', axis=1, inplace=True)
    # Check if 'Total_Stops' column exists in the DataFrame
    if 'Total_Stops' in df.columns:
        df['Total_Stops'] = df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

    # Check if 'Duration' column exists in the DataFrame
    if 'Duration' in df.columns:
        df['Duration_hour'] = df['Duration'].str.split('h').str[0]
        df['Duration_min'] = df['Duration'].str.split('h').str[1].str.replace('m', '')
        df['Duration_min'].fillna(0, inplace=True)
        df['Duration_min'] = df['Duration_min'].astype(int)
        df['Duration_hour'].fillna(0, inplace=True)
        df['Duration_hour'] = df['Duration_hour'].astype(int)
        df.drop('Duration', axis=1, inplace=True)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df['Airline'] = labelencoder.fit_transform(df['Airline'])
    df['Source'] = labelencoder.fit_transform(df['Source'])
    df['Destination'] = labelencoder.fit_transform(df['Destination'])
    df['Additional_Info'] = labelencoder.fit_transform(df['Additional_Info'])
    df['Date'] = df['Date'].astype('int')
    df['Month'] = df['Month'].astype('int')
    df['Year'] = df['Year'].astype('int')
    df['Arrival_min'] = df['Arrival_min'].astype('int')
    df['Arrival_hour'] = df['Arrival_hour'].astype('int')
    df['Dept_hour'] = df['Dept_hour'].astype('int')
    df['Dept_min'] = df['Dept_min'].astype('int')
    
    df.info()
    df.isnull().sum()
    
    return df

# Function to predict flight price
def predict_flight_price(df):
    # Preprocess the input data
    df = preprocess_data(df)
    
    # Convert the preprocessed data into a NumPy array
    input_array = np.array(df).reshape(1, -1)
    
    # Make predictions using the model
    predicted_price = model.predict(input_array)
    
    return predicted_price[0]  # Extract the prediction from the list

# Predict flight price when the user clicks the "Predict" button
if st.button("Predict"):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])
    # Call the prediction function
    predicted_price = predict_flight_price(input_df)

    # Display the predicted price
    st.success(f"Predicted Flight Price: ₹{predicted_price:.2f}")

# Add additional information or explanations to the main app area
st.write("""
## Flight Price Prediction App

This app predicts the price of domestic flights in India based on various input parameters. Adjust the input parameters above and click the "Predict" button to get a price estimate.

The model used for prediction is a Random Forest Regression model trained on historical flight price data.
""")
























