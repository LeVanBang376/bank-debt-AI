import uvicorn
from os import getenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load, dump
from pydantic import BaseModel
from RandomForest import RandomForest
import numpy as np

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    data: list
    
class TrainData(BaseModel):
    xTrain: list
    yTrain: list
    
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

@app.post("/predict", tags=["Root"])
async def hello(data: Data):
    model = None
    try:
        model = load('random_forest_model.joblib')
        result = model.predict(data.data)[0]
        
        return {
            "result": str(result)
        }
    except Exception as e:
        print("error: " + str(e))
        return {
            "error": "Load model failed!"
        }
        
@app.post("/train", tags=["Root"])
async def trainModel(data: TrainData):
    try:
        clf = RandomForest(n_trees=10)
        xTrainData = np.array(data.xTrain)
        yTrainData = np.array(data.yTrain)
        clf.fit(xTrainData, yTrainData)
        dump(clf, 'random_forest_model.joblib')
        return {
            "result": "success"
        }
    except Exception as e:
        print(str(e))
        return {
            "result": "Train model failed!"
        }
    
if __name__ == "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=True)