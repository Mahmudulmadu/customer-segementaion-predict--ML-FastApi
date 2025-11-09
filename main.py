from fastapi import FastAPI
from pydantic import BaseModel, Field, computed_field
from typing import Literal
import pickle
import pandas as pd
import numpy as np


#import the ml model
with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



app = FastAPI()


#pydantic model to validate incoming data

class UserInput(BaseModel):

    Age: float
    Income: float
    Total_Spending: float
    NumWebPurchases: float
    NumStorePurchases: float
    NumWebVisitsMonth: float
    Recency: float



@app.post("/predict-cluster")
def predict_cluster(data: UserInput):
    features = np.array([[data.Age, data.Income, data.Total_Spending, 
                          data.NumWebPurchases, data.NumStorePurchases, 
                          data.NumWebVisitsMonth, data.Recency]])
    
    scaled = scaler.transform(features)
    cluster = int(kmeans.predict(scaled)[0])
    return {"cluster": cluster}