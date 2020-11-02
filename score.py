import json
import numpy as np
import os
import joblib


# Reference - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python#define-an-entry-script

def init():
    global model
    model_path = os.path.join(os.getenv(""), "model.pkl")

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
