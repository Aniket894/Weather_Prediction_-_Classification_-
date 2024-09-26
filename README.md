
# Project Documentation: Weather Classification Prediction

## (ML Classification Project)

## Table of Contents

1. Introduction
2. Dataset Description
3. Project Objectives
4. Project Structure
5. Data Ingestion
6. Data Transformation
7. Model Training
8. Training Pipeline
9. Prediction Pipeline
10. Flask
11. Logging
12. Exception Handling
13. Utils
14. Conclusion


### 1. Introduction
The Weather Classification project aims to classify weather conditions based on various factors such as temperature, humidity, wind speed, and other meteorological metrics. This document provides a comprehensive overview of the project, including its structure, processes, and supporting scripts.

## 2. Dataset Description
Dataset Name: Weather Classification Dataset

### Description: The dataset contains N entries and M columns, providing various features that can help in classifying weather conditions:

- Temperature: The temperature recorded in degrees.
- Humidity: Percentage of humidity in the air.
- Wind Speed: The speed of the wind.
- Wind Direction: Direction from which the wind is coming.
- Precipitation: Amount of precipitation in the area.
- Cloud Cover: Percentage of cloud cover.
- Visibility: How far one can see in the atmosphere.
- Weather Condition: The classification target for the weather condition (e.g., Sunny, Rainy, Snowy, etc.).

## 3. Project Objectives
- **Data Ingestion**: Load and explore the dataset.
- **Data Transformation**: Clean, preprocess, and transform the dataset for analysis.
- **Model Training**: Train various machine learning models to classify weather conditions.
- **Pipeline Creation**: Develop a pipeline for data ingestion, transformation, and model training.
- **Supporting Scripts**: Provide scripts for setup, logging, exception handling, and utilities.

## 4. Project Structure
```
│
├── artifacts/
│   ├── (best)model.pkl
│   ├── LogisticRegression.pkl
│   ├── DecisionTreeClassifier.pkl
│   ├── RandomForestClassifier.pkl
│   ├── GradientBoostingClassifier.pkl
│   ├── XGBoostClassifier.pkl
│   ├── KNeighborsClassifier.pkl
│   ├── raw.csv
│   └── preprocessor.pkl
│
├── notebooks/
│    └──  data/
│         └── weather_classification_data.csv
│  
│
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   ├── weather_icon.png
│   └── style.css
│
├── app.py
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```

## 5. Data Ingestion
The data ingestion module loads the weather dataset, splits it into training and testing sets, and saves them as CSV files. The raw data is stored in the `artifacts/` folder for future reference.

## 6. Data Transformation
The data transformation module handles data preprocessing, including encoding categorical variables (e.g., Weather Condition, Wind Direction) and scaling numerical variables (e.g., Temperature, Humidity, Wind Speed, etc.). The transformed data is stored in the `artifacts/` folder.

## 7. Model Training
The model training module trains multiple machine learning classification models such as:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- KNeighbors Classifier

The best-performing model is saved as `best_model.pkl` in the `artifacts/` folder.

## 8. Training Pipeline
The training pipeline module integrates data ingestion, data transformation, and model training. This ensures that all components are executed in the correct sequence, from loading the data to saving the best model.

## 9. Prediction Pipeline
The prediction pipeline uses the `best_model.pkl` and `preprocessor.pkl` to classify weather conditions on new data. It handles preprocessing and model inference in a seamless manner.

## 10. Flask (app.py)
The Flask app (`app.py`) provides a web interface to input weather data and receive weather classification predictions. The form inputs are collected in `index.html`, and the results are displayed in `results.html`.

![Screenshot 09-26-2024 10 47 11](https://github.com/user-attachments/assets/f03d3120-2809-4004-be89-81df74359144)


![Screenshot 09-26-2024 10 54 53](https://github.com/user-attachments/assets/5d7e9e4d-c1f4-4573-a96a-67c485a8f558)


## 11. Logging
The `logger.py` file captures logs of the project execution, including data ingestion, transformation, model training, and errors encountered. The logs are stored in a designated folder for debugging and monitoring purposes.

## 12. Exception Handling
The `exception.py` file contains the exception handling code, ensuring that any errors in the pipeline are caught and logged. This helps in maintaining the robustness of the project.

## 13. Utils
The `utils.py` file contains utility functions for various repetitive tasks like directory creation, file management, and data loading.

## 14. Conclusion
This documentation outlines the complete workflow of the Weather Classification project, covering the ingestion, transformation, and modeling processes. The project's modular structure allows for easy maintenance, scalability, and adaptability, ensuring that it can be extended for various future use cases.
