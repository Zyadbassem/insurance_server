from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load('model.joblib')

class Features(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age must be between 18 and 120")
    children: int = Field(..., ge=0, description="Cannot have negative children")
    sex: Literal['male', 'female']
    smoker: Literal['yes', 'no']
    region: Literal['southwest', 'southeast', 'northwest', 'northeast']
    bmi: float

    @field_validator('bmi')
    @classmethod
    def check_bmi_range(cls, value):
        if value < 10 or value > 70:
            raise ValueError('BMI must be a realistic value between 10 and 70')
        return value


@app.get('/')
async def root():
    return {"message": "Insurance API is running. Navigate to /docs to test it."}

@app.post('/charge')
async def get_charge(f: Features):
    input_df = pd.DataFrame([f.model_dump()])
    prediction = pipeline.predict(input_df)

    return {'Prediction': float(prediction[0])}
   