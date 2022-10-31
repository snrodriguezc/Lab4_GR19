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
from sklearn.metrics import mean_squared_error as mse

from DataModel import DataModel, DataModelPred, ListModel
from PredictionModel import PredictionModel

app = FastAPI()
model = PredictionModel()

@app.get("/")
def read_root():
   return {"Hello": "World"}

""" @app.post("/predict")
def make_predictions(dataModelPred: DataModelPred):
    print("\n \n \n HERE \n \n \n")
    df = pd.DataFrame(dataModelPred.dict(), columns=dataModelPred.dict().keys(), index=[0])
    df.columns = dataModelPred.columnsPred()
    result = model.make_predictions(df)
    resultList = result.tolist()
    resultJson = json.dumps(resultList)
    return resultJson """

@app.post("/predict")
def make_predictions(listModel: ListModel):
    dict = listModel.dict()
    df = json_normalize(dict['registros'])
    dfpred = DataModelPred
    df.columns = dfpred.columnsPred(dfpred)
    result = model.make_predictions(df)
    resultList = result.tolist()
    resultJson = json.dumps(resultList)
    return resultJson
    

@app.post("/train")
def train(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    
    df_prep = df.copy()
    df_prep = df_prep[['University Rating', 'CGPA', 'Admission Points', 'Research']]

    df_prep = df_prep.drop_duplicates()
    df_prep.duplicated().sum()

    # Volver nulos valores atipicos
    df_prep['Admission Points'] = df_prep['Admission Points'].mask(df_prep['Admission Points'] > 100)
    var_obj = 'Admission Points'
    df_prep = df_prep.dropna(subset=[var_obj])
    
    # Columnas a utilizar para la regresion
    selected_cols = ['CGPA', 'University Rating', 'Research']
    # Se define el transformador que se usará para normalizar las varibales numéricas: MinMaxScaler()
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

    #Column transformer para especificar las transformaciones sobre las columnas seleccionadas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selected_cols)])
    
    pre = [('initial', preprocessor),]
    model = [('model', LinearRegression())]

    # Declara el pipeline utilizando los transformadores (pre) y especificando el modelo o tipo de clasificador (model)
    pipeline = Pipeline(pre+model)  

    df_lineal = df_prep.copy()

     # Extraemos las variables explicativas y objetivo para entrenar
    X = df_lineal.drop(var_obj, axis = 1)
    y = df_lineal[var_obj]

    pipeline = pipeline.fit(X,y)

    r2 = pipeline.score(X,y)

    y_true = y
    y_predicted = pipeline.predict(X)
    
    # Note que hay que sacarle la raiz al valor
    rmse = np.sqrt(mse(y_true, y_predicted))

    # Usamos la lbreria joblib
    filename = 'modelo.joblib'
    # Se guarda
    dump(pipeline, "./assets/"+filename)

    return {
        "R2": r2,
        "RMSE": rmse
    }
