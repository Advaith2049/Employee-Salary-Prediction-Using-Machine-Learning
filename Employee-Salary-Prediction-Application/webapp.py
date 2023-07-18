import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

st.write("""
# Salary Prediction App

This is an salary prediction app that takes in the various inputs as you can see in the sidebar, and 
gives an output estimating the salary. The application has been previously trained on a data set of
more than 10,00,000 features and has been run through various models, finally selecting the one
providing the highest accuracy. 

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Click to View the Data](https://www.kaggle.com/datasets/pavanelisetty/salarypredictions)
""")


def user_input_features():
    job_type = st.sidebar.selectbox('Job Designation', ['ceo','cto','cfo','vice_president','manager','senior','junior','janitor'])
    industry = st.sidebar.selectbox('Industry', ['health', 'web', 'auto', 'finance', 'education', 'oil', 'service'])
    degree = st.sidebar.selectbox('Degree', ['doctoral', 'masters', 'bachelors', 'high_school', 'none'])
    major = st.sidebar.selectbox('Major Subject', ['engineering', 'business', 'literature', 'biology', 'compsci', 'chemistry', 'physics','math','none'])
    exp = st.sidebar.slider('Experience (in Years)', min_value = 0, max_value = 25, step = 1)
    mile = st.sidebar.slider('Miles from a Major Metropolis', min_value = 0, max_value = 100, step = 1)
        
        
    data = {'jobType': job_type,
            'degree': degree,
            'major': major,
            'industry': industry,
            'yearsExperience': exp,
            'milesFromMetropolis': mile}

    features = pd.DataFrame(data, index=[0])
        
    return features
    
input_df = user_input_features()


dataset_raw = pd.read_csv('cleaned_data.csv',index_col = [0])
dataset = dataset_raw.drop(columns=['salary'])
df = pd.concat([input_df,dataset],axis=0)

encode = ['jobType','degree','major','industry']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df.reset_index(drop=True),dummy.reset_index(drop=True)], axis=1)
    del df[col]
df = df[:1]


#Check the Scaling Part
df['yearsExperience'] = df['yearsExperience']/24
df['milesFromMetropolis'] = df['milesFromMetropolis']/99 


st.subheader('User Input features')


st.write(df)

load_clf = pickle.load(open('emp_clf.pkl', 'rb'))


prediction = load_clf.predict(df)

st.subheader('Prediction')
st.write('The predicted salary is $',prediction[0],'K USD per annum')
