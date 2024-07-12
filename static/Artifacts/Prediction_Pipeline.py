import pandas as pd
pd.set_option('display.max_column', 100)
import warnings
warnings.filterwarnings('ignore')

import os 
import pickle
import numpy as np


class HouseData:
    def __init__(
        self,
        area: float,
        bedrooms: int,
        bathrooms: int,
        stories: int,
        mainroad: str,
        guestroom: str,
        basement: str,
        hotwaterheating: str,
        airconditioning: str,
        parking: int,
        prefarea: str,
        furnishingstatus: str
    ):
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.stories = stories
        self.mainroad = mainroad
        self.guestroom = guestroom
        self.basement = basement
        self.hotwaterheating = hotwaterheating
        self.airconditioning = airconditioning
        self.parking = parking
        self.prefarea = prefarea
        self.furnishingstatus = furnishingstatus
    def data_to_dataframe(self):
        data = pd.DataFrame({
            "area":[self.area],
            "bedrooms":[self.bedrooms],
            "bathrooms":[self.bathrooms],
            "stories":[self.stories],
            "mainroad":[self.mainroad],
            "guestroom":[self.guestroom],
            "basement":[self.basement],
            "hotwaterheating":[self.hotwaterheating],
            "airconditioning":[self.airconditioning],
            "parking":[self.parking],
            "prefarea":[self.prefarea],
            "furnishingstatus":[self.furnishingstatus]
        })
        return data

class PredictionPipeline:
    def predict(self, data):
        def load_pickle(path):
            data = None
            with open(path, 'rb') as file:
                data = pickle.load(file)
            return data
        
        model = load_pickle(os.path.join('Artifacts', 'best_model.pkl'))
        transformer = load_pickle(os.path.join('Artifacts', 'transformer.pkl'))
        data = transformer.transform(data)
        predicted_value = model.predict(data)
        return predicted_value