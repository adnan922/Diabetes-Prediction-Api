# Diabetes Prediction Model

This repository contains code and resources for a Diabetes Prediction model using supervised algorithm Support Vector Machine (SVM).

## Overview

The goal of this project is to predict whether a person is diabetic or not based on various features such as pregnancies, glucose level, blood pressure, etc. The model is built using Python and the SKlearn library.

## Dataset

The model is trained on a dataset containing information about individuals, including their medical history and diagnostic measurements. The dataset used can be found in the file `diabetes_dataset.csv`.

## Model Training

The main script `diagnosis.py` contains the code to train the svm model. It uses the `sklearn` library for data preprocessing and for building and training the svm model.

## Usage

1. Make sure you have Python installed. If not, you can download it [here](https://www.python.org/downloads/).
2. Install the required libraries by running:
3. Run the `diabetes_prediction_model.py` script to train the model.

## How to Make Predictions

Once the model is trained, you can make predictions on new data. Use the `data` variable in the script to input new features
```python
# Assuming 'data' is a new set of features for prediction
data = np.array([[...]])  # Replace with actual values

# Make predictions
predictions = model.predict(data)

# Convert probabilities to binary predictions (0 or 1)
binary_predictions = (predictions > 0.5).astype(int)
