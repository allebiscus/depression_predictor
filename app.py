import streamlit as st
import pandas as pd
import joblib

## page configuration
st.set_page_config(
    page_title="Student Depression Risk Assessment",
    layout="wide"
)


## loading the model
model = joblib.load('student_depression_model.pkl')


## main title and description
st.title("🧠 Student Mental Health Risk Assessment")
st.markdown("Please fill out form below to get an assessment of your depression risk level.")


## define options for the form
genders = ['Male', 'Female']
sleep_durations = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
dietary_habits = ['Healthy', 'Moderate', 'Unhealthy']
suicidal_thoughts = ['No', 'Yes']
family_history = ['No', 'Yes']


## input form for assessment
with st.form("assessment_form"):
    
    ## expander for BASIC INFORMATION
    with st.expander("👤 Basic Information", expanded=True):
        ## 2 columns
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 13, 100)
        with col2:
            gender_selected = st.selectbox("Gender", genders)


    ## expander for ACADEMIC AND WORK LIFE
    with st.expander("📚 Academic & Work Life"):
        ## 2 columns
        col1, col2 = st.columns(2)

        with col1:
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 0.01)
            academic_pressure = st.slider("Academic Pressure", 0, 5, help="On a scale of 0 (low) to 5 (high), the level of pressure you face in academic settings.")
            work_pressure = st.slider("Work Pressure", 0, 5, help="On a scale of 0 (low) to 5 (high), the pressure related to work or job responsibilities")

        with col2:
            study_satisfaction = st.slider("Study Satisfaction", 0, 5, help="On a scale of 0 (low) to 5 (high), how satisfied you are with your studies")
            job_satisfaction = st.slider("Job Satisfaction", 0, 4, help="On a scale of 0 (low) to 4 (high), how satisfied you are with your job or work environment")
            work_study_hours = st.slider("Work/Study Hours per Day", 0, 12, help="Average number of hours spent on work or study each day")


    ## expander for LIFESTYLE AND HEALTH
    with st.expander("🏃‍♂️ Lifestyle & Mental Health"):
        ## 2 columns
        col1, col2 = st.columns(2)

        with col1:
            sleep_duration_selected = st.selectbox("Average Sleep Duration", sleep_durations)
            dietary_habits_selected = st.selectbox("Dietary Habits", dietary_habits)
            financial_stress = st.slider("Financial Stress", 1, 5, help="On a scale of 1 (low) to 5 (high), how much stress is experienced due to financial concerns")

        with col2:
            suicidal_thoughts_selected = st.radio("History of Suicidal Thoughts?", suicidal_thoughts, horizontal=True)
            family_history_selected = st.radio("Family History of Mental Illness?", family_history, horizontal=True)


    ## submit button (for form)
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Assess My Risk", type="primary", use_container_width=True)


## prediction + display results
if submitted:
    
    input_data = {
        'Age': [age], 
        'Academic Pressure': [academic_pressure], 
        'Work Pressure': [work_pressure],
        'CGPA': [cgpa], 
        'Study Satisfaction': [study_satisfaction], 
        'Job Satisfaction': [job_satisfaction],
        'Work/Study Hours': [work_study_hours], 
        'Gender': [gender_selected], 
        'Sleep Duration': [sleep_duration_selected],
        'Dietary Habits': [dietary_habits_selected], 
        'Have you ever had suicidal thoughts ?': [suicidal_thoughts_selected],
        'Financial Stress': [financial_stress],
        'Family History of Mental Illness': [family_history_selected]
    }
    df_input = pd.DataFrame(input_data)


    ## OHE for categorical
    categorical_cols = [
        'Gender', 'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'
    ]
    df_input = pd.get_dummies(df_input, columns=categorical_cols)


    ## aligning columns with the model's expected features 
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)


    ## making prediction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]  # probability of the risk score (depressed/not depressed)


    ## display results
    st.markdown("---")
    st.subheader("📊 Assessment Results")
    risk_score = f"{probability:.2%}"

    if prediction == 1:
        st.error("High Risk Detected, Depression", icon="🔴")
    else:
        st.success("Low Risk Detected, No Depression", icon="🟢")

    st.metric(label="Depression Risk Score", value=risk_score)


    ## more detailed analysis
    st.markdown("---")
    st.subheader("🔍 Detailed Analysis from Your Responses")

    col1, col2 = st.columns(2)

    with col1:
        
        ## academic pressure analysis
        if academic_pressure >= 8:
            st.markdown("🔴 **Academic Pressure:** High (High Risk)")

        elif academic_pressure >= 5:
            st.markdown("🟡 **Academic Pressure:** Moderate (Medium Risk)")

        else:
            st.markdown("🟢 **Academic Pressure:** Low (Low Risk)")


        ## work pressure analysis
        if work_pressure >= 8:
            st.markdown("🔴 **Work Pressure:** High (High Risk)")

        elif work_pressure >= 5:
            st.markdown("🟡 **Work Pressure:** Moderate (Medium Risk)")

        else:
            st.markdown("🟢 **Work Pressure:** Low (Low Risk)")
        

        ## sleep analysis
        if sleep_duration_selected == "Less than 5 hours":
            st.markdown("🔴 **Sleep Duration:** Poor (High Risk)")

        elif sleep_duration_selected in ["5-6 hours", "7-8 hours"]:
            st.markdown("🟡 **Sleep Duration:** Suboptimal (Medium Risk)")

        else:
            st.markdown("🟢 **Sleep Duration:** Good (Low Risk)")

        
        ## study satisfaction analysis
        if study_satisfaction <= 3:
            st.markdown("🔴 **Study Satisfaction:** Very Low (High Risk)")

        elif study_satisfaction <= 6:
            st.markdown("🟡 **Study Satisfaction:** Moderate (Medium Risk)")

        else:
            st.markdown("🟢 **Study Satisfaction:** High (Low Risk)")

    with col2:

        ## financial stress analysis
        if financial_stress >= 4:
            st.markdown("🔴 **Financial Stress:** High (High Risk)")

        elif financial_stress >= 3:
            st.markdown("🟡 **Financial Stress:** Moderate (Medium Risk)")

        else:
            st.markdown("🟢 **Financial Stress:** Low (Low Risk)")


        ## suicidal thoughts analysis
        if suicidal_thoughts_selected == "Yes":
            st.markdown("🔴 **Suicidal Thoughts:** Present (High Risk)")

        else:
            st.markdown("🟢 **Suicidal Thoughts:** None (Low Risk)")


        ## family history analysis
        if family_history_selected == "Yes":
            st.markdown("🟡 **Family History:** Present (Medium Risk)")

        else:
            st.markdown("🟢 **Family History:** None (Low Risk)")

## overall styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: 
            linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
            url("https://i.pinimg.com/originals/c8/4f/b7/c84fb740471d58ba9597ace28969d490.gif");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)