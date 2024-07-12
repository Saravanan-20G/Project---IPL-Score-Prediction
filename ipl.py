import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'linear_regression_model.pkl'
model = joblib.load(model_filename)

# Load your dataset for encoding
df = pd.read_csv('C:\\Users\\Saravanan\\OneDrive\\Desktop\\ipl\\ipl.csv')

# One-Hot Encoding on categorical features
df_encoded = pd.get_dummies(df, columns=['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler'])

# Extract original categorical feature columns for Streamlit inputs
original_columns = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']

# Function to predict score
def predict_score(input_data):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    
    # Align the input with the model's training data
    missing_cols = set(df_encoded.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[df_encoded.columns.drop('total')]

    # Make predictions
    prediction = model.predict(input_encoded)[0]
    return prediction

# Streamlit app
def main():
    # Apply custom CSS
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f0f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header image upload
    uploaded_image = st.file_uploader("Upload a header image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, use_column_width=True)
    else:
        st.write("Please upload an image to display at the top of the app.")

    st.title('Cricket Score Prediction')

    # Input fields
    st.subheader('Select Match Details')
    input_data = {}
    for col in original_columns:
        input_data[col] = st.selectbox(f'Select {col.capitalize()}', df[col].unique())

    if st.button('Predict Score'):
        score_prediction = predict_score(input_data)
        st.success(f'Predicted Score: {score_prediction:.2f}')

if __name__ == '__main__':
    main()
