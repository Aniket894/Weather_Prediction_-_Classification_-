from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the pre-trained model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Transform the data using the preprocessor
        processed_data = self.preprocessor.transform(input_df)
        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions.tolist()
    
# Initialize the prediction pipeline with model and preprocessor paths
prediction_pipeline = PredictionPipeline(
    model_path='artifacts/best_model.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    
    input_data = {
        'Temperature': float(request.form.get('Temperature')),
        'Humidity': int(request.form.get('Humidity')),
        'Wind Speed': float(request.form.get('Wind Speed')),
        'Precipitation (%)': float(request.form.get('Precipitation (%)')),
        'Cloud Cover': request.form.get('Cloud Cover'),
        'Atmospheric Pressure': float(request.form.get('Atmospheric Pressure')),
        'UV Index': int(request.form.get('UV Index')),
        'Season': request.form.get('Season'),
        'Visibility (km)': float(request.form.get('Visibility (km)')),
        'Location': request.form.get('Location')
    }

    # Make prediction
    predictions = prediction_pipeline.predict(input_data)
    
    
    
    return render_template('results.html', predictions=predictions)
    
if __name__ == '__main__':
    app.run(debug=True)
