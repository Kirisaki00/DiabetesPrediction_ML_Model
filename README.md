🩺 DiabetesPrediction_ML_Model

A machine learning–based classification system that predicts the likelihood of diabetes using medical diagnostic data. This project demonstrates a complete ML pipeline, including data preprocessing, model training, evaluation, and model persistence.

📌 Project Overview

Diabetes is a chronic disease that requires early detection for effective management. This project uses supervised machine learning techniques to classify whether a patient is diabetic based on key medical attributes.

Key highlights:

Real-world medical dataset

End-to-end ML workflow

Trained and saved classification model

Beginner-friendly and academic-ready

📂 Repository Structure
DiabetesPrediction_ML_Model/
│
├── diabetes.csv                 # Dataset used for training and testing
├── DiabetesPrediction_ML_Model.ipynb               # Jupyter Notebook (EDA, training, evaluation)
├── classification_model.pkl     # Saved trained ML model
├── README.md                    # Project documentation

📊 Dataset Information

The dataset contains several medical predictor variables and one target variable.

🔹 Features

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

🎯 Target Variable

Outcome

1 → Diabetic

0 → Non-diabetic

⚙️ Technologies & Tools

Programming Language: Python

Libraries: NumPy, Pandas, Scikit-learn, Pickle

Environment: Jupyter Notebook

🧠 Machine Learning Pipeline

Load and explore the dataset

Handle missing and invalid values

Split data into training and test sets

Train a classification model

Evaluate performance using accuracy and metrics

Save the trained model for reuse

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Kirisaki00/DiabetesPrediction_ML_Model.git
cd DiabetesDataModel

2️⃣ Install Required Libraries
pip install numpy pandas scikit-learn

3️⃣ Run the Notebook

Open Untitled.ipynb in Jupyter Notebook or JupyterLab and execute the cells in order.

💾 Using the Trained Model

The trained model is saved as:

classification_model.pkl


Load and use it in Python as follows:

import pickle

with open("classification_model.pkl", "rb") as file:
    model = pickle.load(file)


You can then pass new patient data to the model for prediction.

📈 Results

The model achieves good predictive accuracy on test data

Demonstrates effective use of classification algorithms

Suitable for learning, experimentation, and academic evaluation

🚀 Future Enhancements

Add multiple ML models for comparison

Apply hyperparameter tuning

Perform advanced feature scaling

Deploy as a web application (Flask / Streamlit)

Add visual dashboards and reports

🧑‍🎓 Author

Anupam Singh (Kirisaki)
Machine Learning Student Project
