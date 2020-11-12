import json
import numpy as np
import os
import joblib


# Reference - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python#define-an-entry-script

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "outputs/model.pkl")
    #model_path = "./outputs/model.pkl"
    #model_path = "~/cloudfiles/code/Users/odl_user_124394/model.pkl"
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
