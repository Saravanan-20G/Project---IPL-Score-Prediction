 # IPL Score Prediction
 This project aims to predict the total score in an IPL cricket match based on various factors such as venue, batting team, bowling team, batsman, and bowler using machine learning techniques. The project involves data preprocessing, model building, and deployment using Streamlit.
 
 ## Table of Contents
Overview

Dataset

Installation

Usage

Model

Evaluation

Streamlit App

Contributing

License

## Overview
This repository contains code and resources for building a cricket score prediction model. The project uses a linear regression model and a neural network model, both evaluated and compared for performance. The final deployment is done using Streamlit to create an interactive web application for predicting scores.

## Dataset
The dataset used in this project contains IPL match data with features such as date, venue, teams, players, runs, wickets, and overs. The dataset can be found in the ipl_data.csv file.
## Installation
To get started with this project, follow these steps:

Clone the repository:



git clone https://github.com/your-username/cricket-score-prediction.git
cd cricket-score-prediction
Install the required dependencies:



pip install -r requirements.txt
Make sure you have the ipl_data.csv dataset in your project directory.

## Usage
Data Preprocessing and Model Training
Run the data preprocessing and model training script:

bash
Copy code
python model_training.py
This script will preprocess the data, train the linear regression model, and save the trained model as linear_regression_model.pkl.

### Streamlit App
Run the Streamlit app:


streamlit run app.py
Open your web browser and navigate to http://localhost:8501 to interact with the app.

## Model
The project includes two models:

Linear Regression Model: A simple linear regression model trained using the features in the dataset.
Neural Network Model: A neural network model built using Keras and TensorFlow for improved prediction accuracy.
Evaluation
The models are evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared. The evaluation results are printed in the console after running the training script.

## Streamlit App
The Streamlit app provides an interactive interface for users to input match details and get the predicted score. The app takes inputs for the venue, batting team, bowling team, batsman, and bowler, and displays the predicted score.

Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Files in the Repository
ipl_data.csv: The dataset containing IPL match data.
model_training.py: Script for data preprocessing and model training.
app.py: Streamlit app script for score prediction.
requirements.txt: List of required dependencies.
