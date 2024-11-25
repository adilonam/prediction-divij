import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier


# Load your pre-trained model
model = None

# Streamlit app
st.title('📊 Excel File Prediction App')
st.header('0 is negative and 1 is positive')
# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")



if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file , engine='openpyxl')
    
    df['Year'] = df['DATE'].dt.year
    df['Month'] = df['DATE'].dt.month
    df['Day_of_Month'] = df['DATE'].dt.day
    df['Day_of_Week'] = df['DATE'].dt.dayofweek
    df['Day_of_Year'] = df['DATE'].dt.dayofyear
    df['VALUE_is_Positive'] = df['VALUE'].apply(lambda x: 1 if x > 0 else 0)

    df = df.dropna()
    
    # Show the DataFrame
    st.write("DataFrame values:")
    st.dataframe(df)
    
    # Filter for the last week
    plot_data = df[df['DATE'] >= (df['DATE'].max() - pd.Timedelta(days=10))]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.step(plot_data.index, plot_data['VALUE_is_Positive'], where='post', marker='o', linestyle='-', color='b')
    plt.title('Inventory Status Over Time (Last Week)')
    plt.xlabel('Date')
    plt.ylabel('Inventory is Positive')
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(plt)
    
  

    # Drop the 'Date' column
    df_preprocessed = df.drop(columns=['DATE']).dropna()
    
    features = ['Year', 'Month', 'Day_of_Month', 'Day_of_Week', 'Day_of_Year']
    target = 'VALUE_is_Positive'
    # Separate features and target
    X = df_preprocessed[features].values
    y = df_preprocessed[target].values

    # Display shapes on Streamlit
    st.write(f"X shape: {X.shape}")
    st.write(f"y shape: {y.shape}")

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Print progress
    st.write("Training the model...")

    # Build and train the XGBoost model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Print progress
    st.write("Model training completed.")
    
    st.header('Test the accuracy of the model :')
    
    # Make predictions
    y_pred = model.predict(X_test)
    model.fit(X_test, y_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy
    st.write(f"Model Accuracy for day : {accuracy:.2f}")
    
    day_count = st.number_input("Enter the day count", min_value=0, max_value=100, value=6, step=1)
    tolerance = st.number_input("Enter the tolerance", min_value=0, max_value=100, value=2, step=1)
    
    if st.button('Test Accuracy'):
        # Make predictions
        

        # Calculate real and predicted sums for each day_count interval
        real = [y_test[i:i+day_count].sum() for i in range(0, len(y_test), day_count)]
        pr = [y_pred[i:i+day_count].sum() for i in range(0, len(y_pred), day_count)]

        # Calculate the accuracy based on the tolerance of 1
        count = sum(abs(a - b) <= tolerance for a, b in zip(real, pr))
        accuracy_on_days = count / len(real)

        st.write(f"Accuracy on {day_count} days = {accuracy_on_days:.2f}")

    st.header('Predictions :')
    # Date input
    start_date = st.date_input("Select a start date")
    end_date = st.date_input("Select an end date")

    # Create a new DataFrame with all dates between start_date and end_date
    date_range = pd.date_range(start=start_date, end=end_date)
    df_new = pd.DataFrame(date_range, columns=['DATE'])

    # Add the required columns
    df_new['Year'] = df_new['DATE'].dt.year
    df_new['Month'] = df_new['DATE'].dt.month
    df_new['Day_of_Month'] = df_new['DATE'].dt.day
    df_new['Day_of_Week'] = df_new['DATE'].dt.dayofweek
    df_new['Day_of_Year'] = df_new['DATE'].dt.dayofyear

    
    # Get X_scaled using the new DataFrame
    X_new = df_new.drop(columns=['DATE']).values
    X_new_scaled = scaler.transform(X_new)

    if st.button('Predict'):
        
        # Example: Predict using the model
        predictions = model.predict(X_new_scaled)
        
        # Add predictions to the new DataFrame
        df_new['Predicted_Inventory_is_Positive'] = predictions
        
        df_new.drop(columns=['Year', 'Month', 'Day_of_Month', 'Day_of_Week', 'Day_of_Year'], inplace=True)
        df_new['Predicted_Value'] = df_new['Predicted_Inventory_is_Positive'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
        
        # Display the predictions DataFrame
        st.write("Predictions for the selected date range:")
        st.dataframe(df_new)
        st.write(f"Predictions {df_new['Predicted_Inventory_is_Positive'].sum()} Positive and {len(df_new) - df_new['Predicted_Inventory_is_Positive'].sum()} Negative of {len(df_new)} days")
        # Display the prediction with an emoji
        st.write(f"🎉 Predicted values for the selected date range are shown in the table above.")
