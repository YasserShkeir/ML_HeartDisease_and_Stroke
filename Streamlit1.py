import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

# Done by: Yasser Shkeir
# Final Version of the Scikit Model app (V2.0)
# Need to improve UI

#Title of web app + wide layout instead of centered
st.set_page_config(page_title='Heart Disease App', layout='wide')

# Headers in page
st.write("""
# Heart Disease Prediction
# """)

# store a boolean for the heart disease and stroke tests that changes to true only if the user presses the buttons for the tests
if 'bool_heart_disease' not in st.session_state:
    st.session_state['bool_heart_disease']=False

if 'bool_stroke' not in st.session_state:
    st.session_state['bool_stroke']=False

chosen_test = st.selectbox('Please choose the test type:', ('', 'Heart Disease', 'Stroke'))

# Create two Columns for the different test types in the sidebar
col1, col2 = st.columns(2)

with col1:
    #if stroke option is chosen, remove stroke content and display heart disease content
    if chosen_test == 'Heart Disease':
        st.session_state['bool_heart_disease']=True
        st.session_state['bool_stroke']=False

with col2:
    #if stroke option is chosen, remove heart disease content and display stroke content
    if chosen_test == 'Stroke':
        st.session_state['bool_stroke']=True
        st.session_state['bool_heart_disease']=False

def heart_disease_function():
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

    st.subheader('1. Dataset')

    df = pd.read_csv('heart.csv')

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
    if st.sidebar.button('Confirm'):
        msg = 'There is a %' + str(round(rf.predict_proba(predict_df)[0][1] *100, 2)) + ' patient has a heart disease'
        st.error(msg)

def stroke_function():
    test_age=st.sidebar.text_input('Age:', max_chars=3)
    test_sex=st.sidebar.radio('Sex:',options=['Male','Female'])
    test_Hypertension=st.sidebar.select_slider('Hypertension: (0 for no hypertension, 1 for hypertension)',options=[0,1])
    test_Heart_Disease=st.sidebar.select_slider('Heart Disease: (0 for no Heart Disease, 1 for Heart Disease)',options=[0,1])
    test_Ever_Married=st.sidebar.select_slider('Ever Married:', options=['No','Yes'])
    test_Work_Type=st.sidebar.select_slider('Work Type:', options=['Never_worked','children', 'Self-employed', 'Private', 'Govt_job'])
    test_Residence_Type=st.sidebar.select_slider('Residence Type:', options=['Urban', 'Rural'])
    test_Avg_GlucLvl=st.sidebar.text_input('Average Glucose Level:', max_chars=6)
    test_BMI=st.sidebar.text_input('BMI:', max_chars=4)
    test_Smoking_Status=st.sidebar.select_slider('Smoking Status:', options=['Unknown', 'never smoked', 'formerly smoked', 'smokes'])
    predict_data={'age':[test_age],
                'gender':[test_sex],
                'hypertension':[test_Hypertension],
                'heart_disease':[test_Heart_Disease],
                'ever_married':[test_Ever_Married],
                'work_type':[test_Work_Type],
                'Residence_type':[test_Residence_Type],
                'avg_glucose_level':[test_Avg_GlucLvl],
                'bmi':[test_BMI],
                'smoking_status':[test_Smoking_Status]}    

    predict_df= pd.DataFrame(predict_data)

    st.subheader('1. Dataset')

    df = pd.read_csv('stroke-data.csv')
    del df["id"]
    df = df.dropna()
    df = df[df['age'] >= 2]
    df = df[df['gender'] != 'Other']

    st.markdown('**1.1. Glimpse of dataset**')

    st.write(df)

    cleanup_vals = {"gender": {"Male": 0, "Female": 1},
                    "ever_married": {"No": 0, "Yes": 1},
                    "work_type": {"Never_worked": 0, "children": 1, "Self-employed": 2, "Private": 3, "Govt_job": 4},
                    "Residence_type" : {"Urban": 0, "Rural": 1},
                    "smoking_status": {"Unknown": 0, "never smoked": 1, "formerly smoked": 2, "smokes": 3}}

    # df['avg_glucose_level']=df['avg_glucose_level'].apply(lambda x: x * 100)
    # predict_df['avg_glucose_level'].iloc[0]=predict_df['avg_glucose_level'].iloc[0]*100
    # df['bmi']=df['bmi'].apply(lambda x: x * 10)
    # predict_df['bmi']=predict_df['bmi'].apply(lambda x: x * 10)

    df = df.replace(cleanup_vals)
    

    y=df['stroke']
    x=df.drop('stroke',axis=1)

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

    predict_df = predict_df.replace(cleanup_vals)

    st.write('Prediction Score:') 
    if st.sidebar.button('Confirm'):
        msg = 'There is a %' + str(round(rf.predict_proba(predict_df)[0][0] *100, 2)) + ' chance patient will have/had a stroke'
        st.error(msg)

# If the button is pressed, the boolean stored in the session state changes to true, which in this case we call the function
if st.session_state['bool_heart_disease']:
    heart_disease_function()

if st.session_state['bool_stroke']:
    stroke_function()
