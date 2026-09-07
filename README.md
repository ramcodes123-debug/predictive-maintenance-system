# AI Predictive Maintenance System

## 1. Project Overview

This project implements an AI-based predictive maintenance system using machine learning.

The system analyzes industrial machine sensor data and predicts whether a machine is likely to experience a failure.

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Feature selection
- Machine learning model training
- Model evaluation
- Failure prediction
- API-based prediction
- Streamlit application
- Prediction history
- Docker support

---

## 2. Problem Statement

Predictive maintenance is an important application of machine learning in industrial environments.

Traditional maintenance methods are often based on fixed schedules or reactive maintenance after a machine has already failed.

The objective of this project is to develop a machine learning system that can analyze machine operating conditions and predict potential machine failures before they occur.

This can help reduce unexpected downtime, maintenance costs, and operational risks.

---

## 3. Dataset

### AI4I 2020 Predictive Maintenance Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains machine operating and sensor-related information that can be used to predict machine failure.

The dataset includes features such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

The target variable represents whether a machine failure occurred.

The raw dataset is stored in:

DATA/RAW/ai4i2020.csv

---

## 4. Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- XGBoost

### Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Application

- Streamlit

### API

- FastAPI

### Deployment

- Docker

### Development Environment

- JupyterLab

---

## 5. Machine Learning Workflow

The overall workflow is:

Raw Machine Sensor Data
          ↓
Data Loading
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
Feature Selection
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Best Model Selection
          ↓
Prediction
          ↓
API / Streamlit Application

---

## 6. Project Workflow

The overall workflow is:

Raw Machine Sensor Data
          ↓
Data Loading
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
Feature Selection
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Best Model Selection
          ↓
Prediction
          ↓
API / Streamlit Application

---

## 7. Data Preprocessing

The data preprocessing pipeline prepares the raw machine sensor data for machine learning.

The preprocessing process includes:

- Loading the raw dataset
- Removing unnecessary columns
- Checking for missing values
- Checking for duplicate records
- Encoding categorical variables
- Separating input features and target variable
- Splitting the data into training and testing sets
- Scaling numerical features

The processed datasets are stored in:

DATA/PROCESSED/

---

## 8. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the structure and characteristics of the machine data.

The analysis includes:

- Dataset shape and information
- Statistical summary
- Missing-value analysis
- Duplicate-value analysis
- Target variable distribution
- Feature distributions
- Correlation analysis
- Visualization of important machine parameters

The EDA workflow is available in:

NOTEBOOKS/02_eda.ipynb

---

## 9. Feature Engineering

Feature engineering is performed to create useful input features for the machine learning model.

The feature engineering process includes:

- Creating derived features from existing machine parameters
- Transforming relevant numerical variables
- Preparing features for model training
- Maintaining consistency between training and prediction data

The feature engineering implementation is available in:

SRC/feature_engineering.py

---

## 10. Feature Selection

Feature selection is performed to identify the most relevant features for predicting machine failure.

The process helps:

- Remove unnecessary features
- Reduce model complexity
- Improve model efficiency
- Reduce the effect of irrelevant variables
- Prepare the final feature set for model training

The selected features are stored in:

DATA/PROCESSED/selected_features.csv

The feature selection workflow is available in:

NOTEBOOKS/05_feature_selection.ipynb

---

## 11. Model Training

The machine learning model is trained using the processed machine sensor data.

The training process includes:

1. Loading the processed training data
2. Loading the selected features
3. Preparing the target variable
4. Applying feature scaling where required
5. Initializing the machine learning model
6. Training the model
7. Evaluating the model on validation/test data
8. Saving the trained model

The training implementation is available in:

SRC/train.py

The trained model files are stored in:

MODELS/

---

## 12. Model Evaluation

The trained model is evaluated using the test dataset.

The evaluation process includes:

- Test-set prediction
- Accuracy calculation
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Model performance analysis

The evaluation implementation is available in:

SRC/evaluate.py

The evaluation notebook is available in:

NOTEBOOKS/07_evaluation.ipynb

---

## 13. Prediction System

The prediction system accepts machine operating parameters as input and predicts whether a machine failure is likely to occur.

The prediction workflow is:

Machine Sensor Input
          ↓
Input Validation
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
Feature Selection
          ↓
Feature Scaling
          ↓
Trained Model
          ↓
Failure Prediction
          ↓
Prediction Result

The prediction functionality is implemented in:

SRC/predict.py

---

## 14. API

A REST API is provided for machine failure prediction using FastAPI.

The API implementation is located in:

API/main.py

The API accepts machine operating parameters and returns the predicted machine failure result.

The API can be used by external applications or services to obtain predictions from the trained machine learning model.

---

## 15. Streamlit Application

A Streamlit application provides a user-friendly interface for the predictive maintenance system.

The application is implemented in:

APP/app.py

The application allows the user to:

1. Enter machine operating parameters
2. Submit the input values
3. Process the input data
4. Generate a prediction
5. Display the predicted result
6. View the prediction output
7. Store the prediction in the prediction history

---

## 16. Prediction History

Prediction results are stored in the prediction history file.

The prediction history is maintained in:

PREDICTION_HISTORY/prediction_history.json

This file keeps a record of previous prediction results generated by the system.

---

## 17. API Documentation

FastAPI automatically provides interactive API documentation.

After starting the API, the documentation can be accessed at:

http://127.0.0.1:8000/docs

The API documentation allows users to:

- View available API endpoints
- Understand the API request format
- Enter machine sensor values
- Test the prediction endpoint
- View the prediction response

---

## 18. Installation

Clone the repository:

git clone https://github.com/ramcodes123-debug/predictive-maintenance-system.git

Navigate to the project directory:

cd predictive-maintenance-system

Install the required dependencies:

pip install -r requirements.txt

---

## 19. Running the API

The FastAPI application can be started using:

uvicorn API.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation will be available at:

http://127.0.0.1:8000/docs

---

## 20. Running the Streamlit Application

The Streamlit application can be started using:

streamlit run APP/app.py

The application will open in the browser and provide a user-friendly interface for entering machine parameters and generating predictions.

---

## 21. Docker Deployment

The project includes a Dockerfile for containerizing the application.

The Docker configuration is available in:

Dockerfile

Docker can be used to package the application and its dependencies into a consistent environment.

---

## 22. Model Files

The trained machine learning models and supporting preprocessing files are stored in:

MODELS/

The directory contains:

- model.pkl
- predictive_maintenance_model.joblib
- scaler.pkl
- feature_columns.pkl

These files are used during the prediction process.

---

## 23. Results

The model evaluation results are stored in:

DATA/PROCESSED/evaluation_metrics.csv

The evaluation summary is stored in:

DATA/PROCESSED/evaluation_summary.json

The project also generates evaluation outputs and visualizations for analyzing model performance.

---

## 24. Future Improvements

Possible future improvements include:

- Improving prediction accuracy
- Hyperparameter optimization
- Testing additional machine learning algorithms
- Real-time machine sensor integration
- Cloud deployment
- Model monitoring
- Automated model retraining
- Advanced alerting for predicted failures
- Integration with industrial IoT systems

---

## 25. API Request Example

The API can be used by sending the required machine parameters to the prediction endpoint.

Example input:

{
    "air_temp": 305.0,
    "process_temp": 315.0,
    "rot_speed": 1900,
    "torque": 65.0,
    "tool_wear": 280,
    "type": "H"
}

The API processes the input and returns the predicted machine failure result.

---

## 26. Prediction Output

The prediction response provides the machine failure prediction generated by the trained model.

Example output format:

{
    "prediction": "Failure",
    "confidence": "[ADD CONFIDENCE RESULT]"
}

The actual prediction depends on the machine parameters provided as input.

---

## 27. Author

Name: SITARAM

GitHub: https://github.com/ramcodes123-debug

Project Repository: https://github.com/ramcodes123-debug/predictive-maintenance-system.git

---

## 28. License

This project is developed for educational and project submission purposes.

---