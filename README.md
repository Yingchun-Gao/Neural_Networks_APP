# Graduate Admission Prediction (Neural Network)

This project is deployed as an interactive Streamlit application:

https://yingchun-gao-neural-networks-app.streamlit.app/

This application predicts whether a student has a **high chance of graduate admission** based on academic performance and application profile.

A **Neural Network (MLPClassifier)** is trained on historical admission data to classify applicants into two categories:
- High admission chance
- Lower admission chance

## Features

- Complete data preprocessing pipeline
- Binary classification based on admission probability
- Feature scaling using MinMaxScaler
- Neural Network model using MLPClassifier
- Model evaluation using accuracy and confusion matrix
- Feature correlation visualization
- Interactive Streamlit prediction interface

## Dataset

The dataset contains information about graduate school applicants.

Features used for prediction:

- **GRE Score** – Graduate Record Examination score
- **TOEFL Score** – English language proficiency score
- **University Rating** – Rating of undergraduate institution
- **SOP** – Strength of Statement of Purpose
- **LOR** – Strength of Letter of Recommendation
- **CGPA** – Undergraduate GPA
- **Research** – Research experience indicator

During preprocessing:

- **Chance of Admit** is converted into a binary variable:
  - **1** – High chance of admission (≥ 0.8)
  - **0** – Lower chance of admission (< 0.8)

The **Serial_No** column is removed because it has no predictive value.

## Technologies Used

- **Streamlit** – Web application interface
- **Scikit-learn** – Neural network model and evaluation
- **Pandas** – Data preprocessing
- **Matplotlib** – Training visualization
- **Seaborn** – Correlation heatmap visualization