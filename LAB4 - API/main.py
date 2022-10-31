from tokenize import String
import pandas as pd
from pandas import json_normalize
import numpy as np
from joblib import dump, load
from fastapi import FastAPI
import json
from pydantic import Json

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error as mse

from DataModel import DataModelTrain, DataModelPred, ListModelPred, ListModelTrain
from PredictionModel import PredictionModel

app = FastAPI()
model = PredictionModel()

@app.get("/")
def read_root():
   return {"Hello": "World"}

#Endpoint1. Recibe una lista de registros y hace predicciones sobre estos
@app.post("/predict")
def make_predictions(listModelPred: ListModelPred):
    #Transformacion de datos de JSON a dataframe
    dict = listModelPred.dict()
    df = json_normalize(dict['registros'])
    dfpred = DataModelPred
    df.columns = dfpred.columnsPred(dfpred)
    #Predicciones a partir del modelo generado
    result = model.make_predictions(df)
    #Transformacion de datos a JSON
    resultList = result.tolist()
    resultJson = json.dumps(resultList)
    return resultJson
    
@app.post("/train")
def train(listModelTrain: ListModelTrain):
    #Transformacion de datos de JSON a dataframe
    dict = listModelTrain.dict()
    df = json_normalize(dict['registros'])
    dftrain = DataModelTrain
    df.columns = dftrain.columnsTrain(dftrain)
    df_prep = df.copy()
    var_obj = 'Admission Points'
    df_prep = df_prep.dropna(subset=[var_obj])

    #Entrenamiento con datos de entrada
    df_lineal = df_prep.copy()
    X = df_lineal.drop(var_obj, axis = 1)
    y = df_lineal[var_obj]
    model.model.fit(X,y)

    #Generacion de metricas de calidad del modelo
    r2 = model.model.score(X,y)

    y_true = y
    y_predicted = model.model.predict(X)
    
    rmse = np.sqrt(mse(y_true, y_predicted))

    #Almacenamiento del modelo generado
    filename = 'modelo.joblib'
    dump(model.model, "./assets/"+filename)

    return {
        "R2": r2,
        "RMSE": rmse
    }
