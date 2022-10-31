from joblib import load

class PredictionModel:

    #Inicia cargando el modelo generado
    def __init__(self):
        self.model = load("assets/modelo.joblib")

    #Hace las predicciones a partir del modelo generado
    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
