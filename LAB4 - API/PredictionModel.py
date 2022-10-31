from joblib import load

class PredictionModel:

    def __init__(self):
        self.model = load("assets/modelo.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
