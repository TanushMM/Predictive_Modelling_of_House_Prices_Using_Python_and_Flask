from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
pd.set_option('display.max_column', 100)
import warnings
warnings.filterwarnings('ignore')

# Prediction Pipeline Starts here
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
        
        model = load_pickle(os.path.join('static','Artifacts', 'best_model.pkl'))
        transformer = load_pickle(os.path.join('static','Artifacts', 'transformer.pkl'))
        data = transformer.transform(data)
        predicted_value = model.predict(data)
        return predicted_value
# Prediction pipeline Ends here



application = Flask(__name__)

app = application

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    area = request.form['area']
    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']
    stories = request.form['stories']
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    parking = request.form['parking']
    prefarea = request.form['prefarea']
    furnishingstatus = request.form['furnishingstatus']
    house_data = HouseData(float(area), int(bedrooms), int(bathrooms), int(stories),
                            mainroad, guestroom, basement, hotwaterheating,
                            airconditioning, int(parking), prefarea, furnishingstatus)
    house_data = house_data.data_to_dataframe()
    
    prediction_pepiline_object = PredictionPipeline()
    pred_value = prediction_pepiline_object.predict(house_data)
    
    
    return render_template('prediction.html', pred_value=pred_value, input_data=house_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0")