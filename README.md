# Heart Failure Prediction Web Application

This project uses various machine learning models to predict heart failure based on several health factors. It is designed as a web application using the Streamlit library in Python.

[LIVE LINK](https://youtu.be/qsG3wqz_PsI)

## Project Description

Heart failure is a common event caused by Cardiovascular diseases (CVDs), and they are the leading cause of death globally. This web application uses data such as age, anaemia, creatinine phosphokinase levels, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking status, and time (follow up period) to predict the likelihood of heart failure in an individual.

## Installation 

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Streamlit
- XGBoost

To run this application, you'll need to install the necessary libraries. You can install these via pip:

```
pip install -r requirements.txt
```


## Usage

To run the application:

```
streamlit run app.py
```

This command needs to be run in the terminal while in the project directory.

## Model Information

Several machine learning models are used in this project:

- Logistic Regression
- Standard Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boost

The application interface allows users to input their data and then select a machine learning model to make a prediction based on the input data.



