import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

# Done by: Yasser Shkeir
# Final Version of the Scikit Model app (V2.0)
# Need to improve UI

#Title of web app + wide layout instead of centered
st.set_page_config(page_title='Heart Disease and Stroke App', layout='wide')

# Headers in page
st.write("""
# Heart Disease and Stroke Prediction
# """)

# store a boolean for the heart disease and stroke tests that changes to true only if the user presses the buttons for the tests
if 'bool_heart_disease' not in st.session_state:
    st.session_state['bool_heart_disease']=False

if 'bool_stroke' not in st.session_state:
    st.session_state['bool_stroke']=False

# Import Data
df1 = pd.read_csv('heart.csv')


# Create two Columns, use only one to the left for better aesthetics
col1, col2, col3 = st.columns(3)

with col1:
    chosen_test = st.selectbox('Please choose the test type:', ('', 'Heart Disease', 'Stroke'))
    #if none option is chosen, remove all content
    if chosen_test == '':
        st.session_state['bool_heart_disease']=False
        st.session_state['bool_stroke']=False
    #if stroke option is chosen, remove stroke content and display heart disease content
    if chosen_test == 'Heart Disease':
        st.session_state['bool_heart_disease']=True
        st.session_state['bool_stroke']=False
    #if stroke option is chosen, remove heart disease content and display stroke content
    if chosen_test == 'Stroke':
        st.session_state['bool_stroke']=True
        st.session_state['bool_heart_disease']=False   

def heart_disease_function():
    st.subheader('1. Dataset')
    
    # ### Data Cleaning
    new_df = df1[df1['Cholesterol'] != 0]
    new_df = new_df[new_df['RestingBP'] != 0]
    df1['Cholesterol'] = df1['Cholesterol'].replace(0, new_df['Cholesterol'].mean())
    df1['RestingBP'] = df1['RestingBP'].replace(0, new_df['RestingBP'].mean())
    ###

    ### Data Analysis Section ###
    
    df2 = df1

    df2['HeartDisease'] = df2['HeartDisease'].replace([0],'No')
    df2['HeartDisease'] = df2['HeartDisease'].replace([1],'Yes')
    df2['Count'] = 1

    df2 = df2.sort_values(
        by=['HeartDisease'],
        ascending=True).iloc[0:len(df2[df2['HeartDisease']=='No'])*2]

    da_col1, da_col2 = st.columns(2)

    with da_col1:
        st.write('1.1. Dataset Sample')
        st.dataframe(df2.drop('Count', axis=1), height=550)

    with da_col2:
        st.write('1.2. Data Analysis Section')
        selection = st.selectbox('Choose your chart selection:', (df2.columns[0:-2].to_list()))
        if selection != '':
            fig = px.histogram(df2, x=selection, y='Count', title='Count of people with or without a Heart Disease based on {}'.format(selection), color='HeartDisease', barmode='group')
            st.plotly_chart(fig)

    ###

    cleanup_vals = {"Sex": {"M": 0, "F": 1},
                    "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3},
                    "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
                    "ExerciseAngina" : {"Y": 0, "N": 1},
                    "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}}

    df1['Oldpeak']=df1['Oldpeak'].apply(lambda x: x + 2.6)
    df1['Oldpeak']=df1['Oldpeak'].apply(lambda x: x * 10)
    df1 = df1.replace(cleanup_vals)
    df1 = df1.drop('Count', axis=1)

    y=df1['HeartDisease']
    x=df1.drop('HeartDisease',axis=1)

    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3)
    
    # st.markdown('**1.2. Data Splits**')
    # st.write('Training Set')
    # st.info(x_train.shape)

    # st.markdown('**1.3. Variable Details**')
    # st.write('Data Columns')
    # st.info(list(x.columns))

    rf = RandomForestClassifier(n_estimators=250, random_state=0)
    rf.fit(x_train, y_train)

    st.subheader('2. Model ')
    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

    with ml_col1:
        st.write('Length of Training Dataset')
        st.info(len(x_train))

    with ml_col2:
        st.write('Length of Testing Dataset')
        st.info(len(x_test))

    with ml_col3:
        y_pred = rf.predict(x_test)
        st.write('Accuracy Score:')
        st.info(accuracy_score(y_test, y_pred))

    ### INPUT DATA SECTION ###
    st.subheader('3. Data Input Prediction ')
    
    # Create 3 columns for the inputs to be aligned
    with st.form(key='input form'):
        inp_col1, inp_col2, inp_col3 = st.columns(3)
        with inp_col1:
            test_age=st.text_input('Age:', max_chars=3, value=0)
            test_RBP=st.text_input('Resting Blood Pressure:', max_chars=3, value=0)
            test_Cholesterol=st.text_input('Cholesterol:', max_chars=3, value=0)
            test_MHR=st.text_input('Maximum Heart Rate:', max_chars=3, value=0)

        with inp_col2:
            test_CPT=st.select_slider('Chest Pain Type:',options=['TA', 'ATA', 'NAP', 'ASY'])
            test_RECG=st.select_slider('Resting Electrocardiogram:', options=['Normal', 'ST', 'LVH'])
            test_STS=st.select_slider('ST_Slope:', options=['Up', 'Flat', 'Down'])
            test_OPk=st.slider('Old Peak: ', -4.0, 7.0, 0.0, 0.1)

        with inp_col3:
            test_FBS=st.radio('Fasting Blood Sugar:', options=[0,1])
            test_sex=st.radio('Sex:',options=['M','F'])
            test_ExA=st.radio('Exercise Angina:', options=['N','Y'])

        if st.form_submit_button('Confirm'):
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
            # st.write(predict_df)

            predict_df['Oldpeak']=predict_df['Oldpeak'].apply(lambda x: x + 2.6)
            predict_df['Oldpeak']=predict_df['Oldpeak'].apply(lambda x: x * 10)
            predict_df = predict_df.replace(cleanup_vals)
            msg = 'There is a % {} patient has a heart disease'.format(round(rf.predict_proba(predict_df)[0][1] *100, 2))
            st.error(msg)     

def stroke_function():
    df = pd.read_csv('stroke-data.csv')

    st.subheader('1. Dataset')
    
    #### Data Cleaning
    del df["id"]
    df = df.dropna()
    df = df[df['age'] >= 2]
    df = df[df['gender'] != 'Other']
    ###

    df2 = df
    df2['stroke'] = df2['stroke'].replace([0],'No')
    df2['stroke'] = df2['stroke'].replace([1],'Yes')
    df2['Count'] = 1

    df2 = df2.sort_values(
        by=['stroke'],
        ascending=False).iloc[0:len(df2[df2['stroke']=='Yes'])*2]

    da_col1, da_col2 = st.columns(2)

    with da_col1:
        st.write('1.1. Dataset Sample')
        st.dataframe(df2.drop('Count', axis=1), height=550)

    with da_col2:
        st.write('1.2. Data Analysis Section')
        selection = st.selectbox('Choose your chart selection:', (df2.columns[0:-2].to_list()))
        if selection != '':
            fig = px.histogram(df2, x=selection, y='Count', title='Count of people with or without a Stroke based on {}'.format(selection), color='stroke', barmode='group')
            st.plotly_chart(fig)

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
    df = df.drop(columns=['Count'], axis=1)

    y=df['stroke']
    x=df.drop('stroke',axis=1)

    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25)

    rf = RandomForestClassifier(n_estimators=25, random_state=0)
    rf.fit(x_train, y_train)  

    st.subheader('2. Model ')
    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

    with ml_col1:
        st.write('Length of Training Dataset')
        st.info(len(x_train))

    with ml_col2:
        st.write('Length of Testing Dataset')
        st.info(len(x_test))

    with ml_col3:
        y_pred = rf.predict(x_test)
        st.write('Accuracy Score:')
        st.info(accuracy_score(y_test, y_pred))

    ### INPUT DATA SECTION ###
    st.subheader('3. Data Input Prediction ')
    
    # Create 3 columns for the inputs to be aligned
    with st.form(key='input form'):
        inp_col1, inp_col2, inp_col3 = st.columns(3)
        with inp_col1:
            test_age=st.text_input('Age:', max_chars=3, value=0)
            test_Avg_GlucLvl=st.text_input('Average Glucose Level:', max_chars=6, value=0)
            test_BMI=st.text_input('BMI:', max_chars=4, value=0)

        with inp_col2:
            test_Hypertension=st.select_slider('Hypertension: (0 for no hypertension, 1 for hypertension)',options=[0,1])
            test_Heart_Disease=st.select_slider('Heart Disease: (0 for no Heart Disease, 1 for Heart Disease)',options=[0,1])
            test_Work_Type=st.select_slider('Work Type:', options=['Never_worked','children', 'Self-employed', 'Private', 'Govt_job'])
            test_Smoking_Status=st.select_slider('Smoking Status:', options=['Unknown', 'never smoked', 'formerly smoked', 'smokes'])

        with inp_col3:
            test_sex=st.radio('Sex:',options=['Male','Female'])
            test_Residence_Type=st.radio('Residence Type:', options=['Urban', 'Rural'])
            test_Ever_Married=st.radio('Ever Married:', options=['No','Yes'])

        if st.form_submit_button('Confirm'):     
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

            predict_df = predict_df.replace(cleanup_vals)

            msg = 'There is a % {} chance patient will have/had a stroke'.format(round(rf.predict_proba(predict_df)[0][0] *100, 2))
            st.error(msg)

# If the button is pressed, the boolean stored in the session state changes to true, which in this case we call the function
if st.session_state['bool_heart_disease']:
    heart_disease_function()

if st.session_state['bool_stroke']:
    stroke_function()