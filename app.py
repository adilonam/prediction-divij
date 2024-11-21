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

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")



if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file , engine='openpyxl')
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day_of_Month'] = df['Date'].dt.day
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Inventory_is_Positive'] = df['Inventory'].apply(lambda x: 1 if x > 0 else 0)

    df = df.dropna()
    
    # Show the DataFrame
    st.write("DataFrame values:")
    st.dataframe(df)
    
    # Filter for the last week
    plot_data = df[df['Date'] >= (df['Date'].max() - pd.Timedelta(days=10))]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.step(plot_data.index, plot_data['Inventory_is_Positive'], where='post', marker='o', linestyle='-', color='b')
    plt.title('Inventory Status Over Time (Last Week)')
    plt.xlabel('Date')
    plt.ylabel('Inventory is Positive')
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(plt)
    
  

    # Drop the 'Date' column
    df_preprocessed = df.drop(columns=['Date'])

    # Separate features and target
    X = df_preprocessed.drop(columns=['Inventory_is_Positive' , "Inventory"]).values
    y = df_preprocessed['Inventory_is_Positive'].values

    # Display shapes on Streamlit
    st.write(f"X shape: {X.shape}")
    st.write(f"y shape: {y.shape}")

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # Print progress
    st.write("Training the model...")

    # Build and train the XGBoost model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Print progress
    st.write("Model training completed.")

    # Make predictions
    y_pred = model.predict(X_test)
    model.fit(X_test, y_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy
    # st.write(f"Model Accuracy: {accuracy:.2f}")

   
    # Date input
    start_date = st.date_input("Select a start date")
    end_date = st.date_input("Select an end date")

    # Create a new DataFrame with all dates between start_date and end_date
    date_range = pd.date_range(start=start_date, end=end_date)
    df_new = pd.DataFrame(date_range, columns=['Date'])

    # Add the required columns
    df_new['Year'] = df_new['Date'].dt.year
    df_new['Month'] = df_new['Date'].dt.month
    df_new['Day_of_Month'] = df_new['Date'].dt.day
    df_new['Day_of_Week'] = df_new['Date'].dt.dayofweek
    df_new['Day_of_Year'] = df_new['Date'].dt.dayofyear

    # Get X_scaled using the new DataFrame
    X_new = df_new.drop(columns=['Date']).values
    X_new_scaled = scaler.transform(X_new)

    if st.button('Predict'):
        # Example: Predict using the model
        predictions = model.predict(X_new_scaled)
        
        # Add predictions to the new DataFrame
        df_new['Predicted_Inventory_is_Positive'] = predictions
        
        df_new.drop(columns=['Year', 'Month', 'Day_of_Month', 'Day_of_Week', 'Day_of_Year'], inplace=True)
        
        # Display the predictions DataFrame
        st.write("Predictions for the selected date range:")
        st.dataframe(df_new)
        
        # Display the prediction with an emoji
        st.write(f"🎉 Predicted values for the selected date range are shown in the table above.")
