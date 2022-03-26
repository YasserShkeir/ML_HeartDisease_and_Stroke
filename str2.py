import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title='Heart Disease App', layout='wide')

st.sidebar.write('# Choose the test type:')

col1, col2 = st.sidebar.columns(2)

with col1:
    st.button('* Heart Disease *')

with col2:
    st.button('* * * Stroke * * *')

st.sidebar.header('Please add your test results')
test_age=st.sidebar.text_input('Age:', max_chars=3)
test_sex=st.sidebar.radio('Sex:',options=['M','F'])
test_CPT=st.sidebar.select_slider('Chest Pain Type:',options=['TA', 'ATA', 'NAP', 'ASY'])
test_RBP=st.sidebar.text_input('Resting Blood Pressure:', max_chars=3)
test_Cholesterol=st.sidebar.text_input('Cholesterol:', max_chars=3)
test_FBS=st.sidebar.select_slider('Fasting Blood Sugar:', options=[0,1])
test_RECG=st.sidebar.select_slider('Resting Electrocardiogram:', options=['Normal', 'ST', 'LVH'])
test_MHR=st.sidebar.text_input('Maximum Heart Rate:', max_chars=3)
test_ExA=st.sidebar.radio('Exercise Angina:', options=['N','Y'])
test_OPk=st.sidebar.slider('Old Peak: ', -4.0, 7.0, 0.0, 0.1)
test_STS=st.sidebar.select_slider('ST_Slope:', options=['Up', 'Flat', 'Down'])

predict_data={'Age':[test_age],
                'Sex':[test_sex],
                'ChestPainType':[test_CPT],
                'RestingBP':[test_RBP],
                'Cholesterol':[test_Cholesterol],
                'FastingBS':[test_FBS],
                'RestingECG':[test_RECG],
                'MaxHR':[test_MHR],
                'ExerciseAngina':[test_ExA],
                'Oldpeak':[test_OPk],
                'ST_Slope':[test_STS]}    

predict_df= pd.DataFrame(predict_data)



st.write("""
# Heart Disease Prediction
# 
# Random Forest Classifier Approach
# """)

st.subheader('1. Dataset')

df = pd.read_csv('Part 1 HeartDisease/heart.csv')

st.markdown('**1.1. Glimpse of dataset**')

st.write(df)

cleanup_vals = {"Sex":     {"M": 0, "F": 1},
                "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3},
                "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
                "ExerciseAngina" : {"Y": 0, "N": 1},
                "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}}

df['Oldpeak']=df['Oldpeak'].apply(lambda x: x + 2.6)
df['Oldpeak']=df['Oldpeak'].apply(lambda x: x * 10)
predict_df['Oldpeak']=predict_df['Oldpeak'].apply(lambda x: x + 2.6)
predict_df['Oldpeak']=predict_df['Oldpeak'].apply(lambda x: x * 10)

df = df.replace(cleanup_vals)
predict_df = predict_df.replace(cleanup_vals)

y=df['HeartDisease']
x=df.drop('HeartDisease',axis=1)

# width = st.slider("plot width", 5, 20, 5)
# height = st.slider("plot height", 5, 20, 5)

# colormap = np.array(['r','b']) # for marking data with or without heartdisease on scatter plot
# m = df['Oldpeak']
# n = df['MaxHR']
# d = df['HeartDisease'] # used to color plots with Heartdisease in red
# fig, ax1 = plt.subplots(figsize=(width, height))
# ax2 = ax1.twinx()
# ax1.scatter(m, n, c=colormap[d])
# ax2.plot(np.sort(m), np.arange(m.size), c='r')

# st.pyplot(fig)

st.markdown('**1.2. Data Splits**')
st.write('Training Set')
st.info(x.shape)

st.markdown('**1.3. Variable Details**')
st.write('Data Columns')
st.info(list(x.columns))

x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3)

rf = RandomForestClassifier(n_estimators=250, random_state=0)
rf.fit(x_train, y_train)

st.subheader('2. Model Performance')

y_pred = rf.predict(x_test)
st.write('Accuracy Score:')
st.info(accuracy_score(y_test, y_pred))

st.write('Prediction Score:') 
st.write('[[No Disease - Disease]]') 

if st.sidebar.button('Confirm'):
    st.warning(rf.predict_proba(predict_df))