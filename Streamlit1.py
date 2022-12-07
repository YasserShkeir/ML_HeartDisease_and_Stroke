import pandas as pd
import streamlit as st
import plotly.express as px
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

# Done by: Yasser Shkeir
# Final Version of the Scikit Model app (V2.0)
# Need to improve UI

# Title of web app + wide layout instead of centered
st.set_page_config(page_title="Heart Disease and Stroke App", layout="wide")

# Headers in page
st.write(
    """
# Heart Disease and Stroke Prediction
# """
)

# store a boolean for the heart disease and stroke tests that changes to true only if the user presses the buttons for the tests
if "bool_heart_disease" not in st.session_state:
    st.session_state["bool_heart_disease"] = False

if "bool_stroke" not in st.session_state:
    st.session_state["bool_stroke"] = False

# Create two Columns, use only one to the left for better aesthetics
col1, col2, col3 = st.columns(3)

with col1:
    chosen_test = st.selectbox(
        "Please choose the test type:", ("", "Heart Disease", "Stroke")
    )
    # if none option is chosen, remove all content
    if chosen_test == "":
        st.session_state["bool_heart_disease"] = False
        st.session_state["bool_stroke"] = False
    # if stroke option is chosen, remove stroke content and display heart disease content
    if chosen_test == "Heart Disease":
        st.session_state["bool_heart_disease"] = True
        st.session_state["bool_stroke"] = False
    # if stroke option is chosen, remove heart disease content and display stroke content
    if chosen_test == "Stroke":
        st.session_state["bool_stroke"] = True
        st.session_state["bool_heart_disease"] = False

df_heart = pd.read_csv("heart.csv")

# ### Data Cleaning
new_df = df_heart[df_heart["Cholesterol"] != 0]
new_df = new_df[new_df["RestingBP"] != 0]
df_heart["Cholesterol"] = df_heart["Cholesterol"].replace(
    0, new_df["Cholesterol"].mean()
)
df_heart["Cholesterol"] = df_heart["Cholesterol"].apply(
    lambda x: x + random.randrange(-50, 50)
)
df_heart["RestingBP"] = df_heart["RestingBP"].replace(0, new_df["RestingBP"].mean())

cleanup_vals = {
    "Sex": {"M": 0, "F": 1},
    "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3},
    "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
    "ExerciseAngina": {"Y": 0, "N": 1},
    "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2},
}

df_heart["Oldpeak"] = df_heart["Oldpeak"].apply(lambda x: x + 2.6)
df_heart["Oldpeak"] = df_heart["Oldpeak"].apply(lambda x: x * 10)
df_heart = df_heart.replace(cleanup_vals)

y = df_heart["HeartDisease"]
x = df_heart.drop("HeartDisease", axis=1)

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33)

rf_heart = RandomForestClassifier(n_estimators=25, random_state=0)
rf_heart.fit(x_train, y_train)


def heart_disease_function():
    df = pd.read_csv("heart.csv")

    st.subheader("1. Dataset")

    # ### Data Cleaning
    new_df = df[df["Cholesterol"] != 0]
    new_df = new_df[new_df["RestingBP"] != 0]
    df["Cholesterol"] = df["Cholesterol"].replace(0, new_df["Cholesterol"].mean())
    df["Cholesterol"] = df["Cholesterol"].apply(lambda x: x + random.randrange(-50, 50))
    df["RestingBP"] = df["RestingBP"].replace(0, new_df["RestingBP"].mean())
    ###

    ### Data Analysis Section ###

    df2 = df

    df2["HeartDisease"] = df2["HeartDisease"].replace([0], "No")
    df2["HeartDisease"] = df2["HeartDisease"].replace([1], "Yes")
    df2["Count"] = 1

    df2 = df2.sort_values(by=["HeartDisease"], ascending=True).iloc[
        0 : len(df2[df2["HeartDisease"] == "No"]) * 2
    ]

    da_col1, da_col2 = st.columns(2)

    with da_col1:
        st.write("1.1. Dataset Sample")
        st.dataframe(df2.drop("Count", axis=1), height=550)

    with da_col2:
        st.write("1.2. Data Analysis Section")
        selection = st.selectbox(
            "Choose your chart selection:", (df2.columns[0:-2].to_list())
        )
        if selection != "":
            fig = px.histogram(
                df2,
                x=selection,
                y="Count",
                title="Count of people with or without a Heart Disease based on {}".format(
                    selection
                ),
                color="HeartDisease",
                barmode="group",
            )
            st.plotly_chart(fig)

    ###

    df = df.drop("Count", axis=1)

    # st.markdown('**1.2. Data Splits**')
    # st.write('Training Set')
    # st.info(x_train.shape)

    # st.markdown('**1.3. Variable Details**')
    # st.write('Data Columns')
    # st.info(list(x.columns))

    st.subheader("2. Model ")
    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

    with ml_col1:
        st.write("Length of Training Dataset")
        st.info(len(x_train))

    with ml_col2:
        st.write("Length of Testing Dataset")
        st.info(len(x_test))

    with ml_col3:
        y_pred = rf_heart.predict(x_test)
        st.write("Accuracy Score:")
        st.info(accuracy_score(y_test, y_pred))

    ### INPUT DATA SECTION ###
    st.subheader("3. Data Input Prediction ")

    # Create 3 columns for the inputs to be aligned
    with st.form(key="input form"):
        inp_col1, inp_col2, inp_col3 = st.columns(3)
        with inp_col1:
            test_age = st.text_input("Age:", max_chars=3, value=0)
            test_RBP = st.text_input("Resting Blood Pressure:", max_chars=3, value=0)
            test_Cholesterol = st.text_input("Cholesterol:", max_chars=3, value=0)
            test_MHR = st.text_input("Maximum Heart Rate:", max_chars=3, value=0)

        with inp_col2:
            test_CPT = st.select_slider(
                "Chest Pain Type:", options=["TA", "ATA", "NAP", "ASY"]
            )
            test_RECG = st.select_slider(
                "Resting Electrocardiogram:", options=["Normal", "ST", "LVH"]
            )
            test_STS = st.select_slider("ST_Slope:", options=["Up", "Flat", "Down"])
            test_OPk = st.slider("Old Peak: ", -4.0, 7.0, 0.0, 0.1)

        with inp_col3:
            test_FBS = st.radio("Fasting Blood Sugar:", options=[0, 1])
            test_sex = st.radio("Sex:", options=["M", "F"])
            test_ExA = st.radio("Exercise Angina:", options=["N", "Y"])

        if st.form_submit_button("Confirm"):
            predict_data = {
                "Age": [test_age],
                "Sex": [test_sex],
                "ChestPainType": [test_CPT],
                "RestingBP": [test_RBP],
                "Cholesterol": [test_Cholesterol],
                "FastingBS": [test_FBS],
                "RestingECG": [test_RECG],
                "MaxHR": [test_MHR],
                "ExerciseAngina": [test_ExA],
                "Oldpeak": [test_OPk],
                "ST_Slope": [test_STS],
            }
            predict_df = pd.DataFrame(predict_data)
            # st.write(predict_df)

            predict_df["Oldpeak"] = predict_df["Oldpeak"].apply(lambda x: x + 2.6)
            predict_df["Oldpeak"] = predict_df["Oldpeak"].apply(lambda x: x * 10)
            predict_df = predict_df.replace(cleanup_vals)
            msg = "There is a % {} patient has a heart disease".format(
                round(rf_heart.predict_proba(predict_df)[0][1] * 100, 2)
            )
            st.error(msg)


def stroke_function():
    df = pd.read_csv("stroke-data.csv")

    st.subheader("1. Dataset")

    #### Data Cleaning
    del df["id"]
    df = df.dropna()
    df = df[df["age"] >= 2]
    df = df[df["gender"] != "Other"]
    ###

    df2 = df
    df2["stroke"] = df2["stroke"].replace([0], "No")
    df2["stroke"] = df2["stroke"].replace([1], "Yes")
    df2["Count"] = 1

    df2 = df2.sort_values(by=["stroke"], ascending=False).iloc[
        0 : len(df2[df2["stroke"] == "Yes"]) * 2
    ]

    da_col1, da_col2 = st.columns(2)

    with da_col1:
        st.write("1.1. Dataset Sample")
        st.dataframe(df2.drop("Count", axis=1), height=550)

    with da_col2:
        st.write("1.2. Data Analysis Section")
        selection = st.selectbox(
            "Choose your chart selection:", (df2.columns[0:-2].to_list())
        )
        if selection != "":
            fig = px.histogram(
                df2,
                x=selection,
                y="Count",
                title="Count of people with or without a Stroke based on {}".format(
                    selection
                ),
                color="stroke",
                barmode="group",
            )
            st.plotly_chart(fig)

    cleanup_vals = {
        "gender": {"Male": 0, "Female": 1},
        "ever_married": {"No": 0, "Yes": 1},
        "work_type": {
            "Never_worked": 0,
            "children": 1,
            "Self-employed": 2,
            "Private": 3,
            "Govt_job": 4,
        },
        "Residence_type": {"Urban": 0, "Rural": 1},
        "smoking_status": {
            "Unknown": 0,
            "never smoked": 1,
            "formerly smoked": 2,
            "smokes": 3,
        },
    }

    df = df.replace(cleanup_vals)
    df = df.drop(columns=["Count"], axis=1)

    y = df["stroke"]
    x = df.drop("stroke", axis=1)

    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33)

    rf_stroke = RandomForestClassifier(n_estimators=100)

    rf_stroke.fit(x_train, y_train)

    st.subheader("2. Model Training")

    ml_col1, ml_col2, ml_col3 = st.columns(3)

    with ml_col1:
        st.write("2.1. Model Training")
        st.info("Random Forest Classifier")

    with ml_col2:
        st.write("2.2. Model Accuracy")
        y_pred = rf_stroke.predict(x_test)
        st.info(accuracy_score(y_test, y_pred))

    with ml_col3:
        st.write("2.3. Model Parameters")
        st.info(rf_stroke.get_params())

    st.subheader("3. Model Prediction")

    with st.form(key="stroke_form"):

        inp_col1, inp_col2, inp_col3 = st.columns(3)

        with inp_col1:
            test_age = st.text_input("Age:", max_chars=3, value=0)
            test_Avg_GlucLvl = st.text_input(
                "Average Glucose Level:", max_chars=6, value=0
            )
            test_BMI = st.text_input("BMI:", max_chars=4, value=0)

        with inp_col2:
            test_Hypertension = st.radio("Hypertension:", options=[0, 1])
            test_heart_disease = st.radio("Heart Disease:", options=[0, 1])
            test_Work_Type = st.selectbox(
                "Work Type:",
                options=[
                    "Never_worked",
                    "children",
                    "Self-employed",
                    "Private",
                    "Govt_job",
                ],
            )
            test_Smoking_Status = st.selectbox(
                "Smoking Status:",
                options=["Unknown", "never smoked", "formerly smoked", "smokes"],
            )

        with inp_col3:
            test_sex = st.radio("Sex:", options=["Male", "Female"])
            test_Residence_Type = st.radio(
                "Residence Type:", options=["Urban", "Rural"]
            )
            test_married = st.radio("Married:", options=["No", "Yes"])

        submit_button = st.form_submit_button(label="Predict")

        if submit_button:
            predict_data = {
                "age": [test_age],
                "gender": [test_sex],
                "hypertension": [test_Hypertension],
                "heart_disease": [test_heart_disease],
                "ever_married": [test_married],
                "work_type": [test_Work_Type],
                "Residence_type": [test_Residence_Type],
                "avg_glucose_level": [test_Avg_GlucLvl],
                "bmi": [test_BMI],
                "smoking_status": [test_Smoking_Status],
            }

            predict_df = pd.DataFrame(predict_data)

            predict_df = predict_df.replace(cleanup_vals)
            msg = "There is a % {} patient has a stroke".format(
                round(rf_stroke.predict_proba(predict_df)[0][1] * 100, 2)
            )
            st.error(msg)


# If the button is pressed, the boolean stored in the session state changes to true, which in this case we call the function
if st.session_state["bool_heart_disease"]:
    heart_disease_function()

if st.session_state["bool_stroke"]:
    stroke_function()
