import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Exam Score Predictor", layout="centered")

@st.cache_resource
def load_data_and_train():
    data = pd.read_csv("Exam_Score_Prediction.csv")
    
    cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le
    
    features = ['age', 'gender', 'course', 'study_hours', 'class_attendance', 'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
    X = data[features]
    y = data['exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=1234)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return model, le_dict, features, round(r2, 4)

st.title("🎓 Exam Score Prediction Dashboard")
st.write("Apne details enter karein aur dekhein aapka predicted exam score kya hoga.")

with st.spinner("Loading model..."):
    model, le_dict, features, r2_score = load_data_and_train()

st.sidebar.header("Student Input Features")

age = st.sidebar.slider('Age', 17, 30, 20)
study_hours = st.sidebar.number_input('Daily Study Hours', 0.0, 24.0, 5.0)
class_attendance = st.sidebar.slider('Class Attendance (%)', 0, 100, 75)
sleep_hours = st.sidebar.number_input('Sleep Hours', 0.0, 12.0, 7.0)
gender = st.sidebar.selectbox('Gender', ['male', 'female', 'other'])
course = st.sidebar.selectbox('Course', ['diploma', 'bca', 'b.sc', 'b.tech', 'bba', 'ba', 'b.com'])
internet_access = st.sidebar.selectbox('Internet Access', ['yes', 'no'])
sleep_quality = st.sidebar.selectbox('Sleep Quality', ['poor', 'average', 'good'])
study_method = st.sidebar.selectbox('Study Method', ['coaching', 'online videos', 'mixed', 'self-study', 'group study'])
facility_rating = st.sidebar.selectbox('Facility Rating', ['low', 'medium', 'high'])
exam_difficulty = st.sidebar.selectbox('Exam Difficulty', ['hard', 'moderate', 'easy'])

with st.sidebar:
    st.divider()
    st.success(f"Model R2 Score: {r2_score:.4f}")
    st.caption("Dataset: 20,000 students")

if st.button('Predict Score', use_container_width=True, type="primary"):
    data = {
        'age': age, 'gender': gender, 'course': course, 'study_hours': study_hours,
        'class_attendance': class_attendance, 'internet_access': internet_access,
        'sleep_hours': sleep_hours, 'sleep_quality': sleep_quality, 'study_method': study_method,
        'facility_rating': facility_rating, 'exam_difficulty': exam_difficulty
    }
    df = pd.DataFrame([data])
    for col in ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']:
        df[col] = le_dict[col].transform(df[col])
    prediction = model.predict(df)[0]
    st.divider()
    st.success(f"🎯 Aapka Predicted Exam Score hai: **{prediction:.2f}**")
    st.progress(min(int(prediction), 100))
