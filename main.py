# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:31:31 2023

@author: CUK
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app =FastAPI()

class model_input(BaseModel):
    
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    

diabetes_model= pickle.load(open('train_model.sav', 'rb')) 


@app.post('/diabetes_prediction')
def diabites_pred(input_parameters: model_input):
    
    input_data= input_parameters.json()
    input_dictionary= json.loads(input_data)
    
    preg= input_dictionary['Pregnancies']
    giu= input_dictionary['Glucose']
    bp= input_dictionary['BloodPressure']
    skin= input_dictionary['SkinThickness']
    insulin= input_dictionary['Insulin']
    BMI= input_dictionary['BMI']
    dpf= input_dictionary['DiabetesPedigreeFunction']
    age= input_dictionary['Age']
    
    input_list= [preg, giu, bp, skin, insulin, BMI, dpf, age]
    
    
    prediction= diabetes_model.predict([input_list])
    
    if prediction[0] == 0:
        return 'The Person is not Diabetic'
    else:
        return 'The Person in Diabetic'


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
    
    
    
    
    
    
    
    
