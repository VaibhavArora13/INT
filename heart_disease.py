import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
 

heart_disease_model = pickle.load(open('model.pkl', 'rb'))


# page title
st.title("Heart Disease Prediction using ML")
st.image("""https://drive.google.com/uc?export=view&id=1j70ye8HLzzjYx8-T0R8485V24xlCmg9S""")
st.header('Enter the Features of the Heart Disease:')

age = st.number_input('Age:', min_value=1, max_value=100)

sex = st.selectbox('Gender:', ['Male', 'Female'])

cp = st.selectbox('Type of chest pain experienced:', ['typical angina', 'atypical angina', 'non- anginal pain', 'asymptomatic'])

trestbps = st.number_input('level of blood pressure at resting mode in mm/HG:', min_value=94, max_value=200)

chol = st.number_input('Serum cholesterol in mg/dl:', min_value=126, max_value=564)

fbs = st.selectbox('Blood sugar levels on fasting > 120 mg/dl:', ['False', 'True'])

restecg = st.selectbox('Result of electrocardiogram while at rest:', ['Normal', 'ST-T wave abnormality', 'definite left ventricular hypertrophy'])

thalach = st.number_input('Maximum heart rate achieved:', min_value=71, max_value=202)

exang = st.selectbox('Angina induced by exercise:', ['false', 'true'])

oldpeak = st.number_input('Exercise induced ST-depression while at rest:', min_value=0.0, max_value=6.2)
 
slope = st.selectbox('ST segment during peak exercise:', ['up sloping', 'flat', 'down sloping'])

ca = st.selectbox('The number of major vessels:', ['One', 'Two', 'Three'])

thal = st.selectbox('thalassemia:', ['Null', 'normal blood flow', 'fixed defect', 'reversible defect'])


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):

    if sex == 'Male':
        sex = 1
    elif sex == 'Female':
        sex = 0
    
    if cp == 'typical angina':
        cp = 0
    elif cp == 'atypical angina':
        cp = 1
    elif cp == 'non- anginal pain':
        cp = 2
    elif cp == 'asymptomatic':
        cp = 3
    
    if fbs == 'False':
        fbs = 0
    elif fbs == 'True':
        fbs = 1

    if restecg == 'Normal':
        restecg = 0
    elif restecg == 'ST-T wave abnormality':
        restecg = 1
    elif restecg == 'definite left ventricular hypertrophy':
        restecg = 2

    if exang == 'false':
        exang = 0
    elif exang == 'true':
        exang = 1

    if slope == 'up sloping':
        slope = 0
    elif slope == 'flat':
        slope = 1
    elif slope == 'down sloping':
        slope = 2

    if ca == 'Zero':
        ca = 0
    elif ca == 'One':
        ca = 1
    elif ca == 'Two':
        ca = 2
    elif ca == 'Three':
        ca = 3

    if thal == 'Null':
        thal = 0
    elif thal == 'normal blood flow':
        thal = 1
    elif thal == 'fixed defect':
        thal = 2
    elif thal == 'reversible defect':
        thal = 3

    data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    for col in ['thalach', 'chol', 'trestbps', 'oldpeak']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        data[col] = data[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

    mms = MinMaxScaler() 
    ss = StandardScaler() 
    data['oldpeak'] = mms.fit_transform(data[['oldpeak']])
    data['age'] = ss.fit_transform(data[['age']])
    data['trestbps'] = ss.fit_transform(data[['trestbps']])
    data['chol'] = ss.fit_transform(data[['chol']])
    data['thalach'] = ss.fit_transform(data[['thalach']])

    prediction = heart_disease_model.predict(data)
    return prediction

if st.button('Predict Heart Disease'):
    Results = predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if Results[0] == 0:
        st.success("Hurray you don't have any Heart Disease")
    elif Results[0] == 1:
        st.success("OOPS you have heart disease")