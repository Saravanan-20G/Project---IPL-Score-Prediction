import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model_filename = 'linear_regression_model.pkl'
model = joblib.load(model_filename)

# Load your dataset
df = pd.read_csv('your_dataset.csv')  # Update with your actual dataset path

# Function to predict score
def predict_score(venue, bat_team, bowl_team, batsman, bowler):
    # Prepare input features as a DataFrame
    input_df = pd.DataFrame({
        'venue': [venue],
        'bat_team': [bat_team],
        'bowl_team': [bowl_team],
        'batsman': [batsman],
        'bowler': [bowler]
    })

    # Perform any preprocessing necessary on input_df (like label encoding)

    # Make predictions
    prediction = model.predict(input_df)

    return prediction

# Streamlit app
def main():
    st.title('Cricket Score Prediction')

    # Input fields
    venue = st.selectbox('Venue', df['venue'].unique())
    bat_team = st.selectbox('Batting Team', df['bat_team'].unique())
    bowl_team = st.selectbox('Bowling Team', df['bowl_team'].unique())
    batsman = st.selectbox('Batsman', df['batsman'].unique())
    bowler = st.selectbox('Bowler', df['bowler'].unique())

    if st.button('Predict Score'):
        score_prediction = predict_score(venue, bat_team, bowl_team, batsman, bowler)
        st.success(f'Predicted Score: {score_prediction}')

if __name__ == '__main__':
    main()
