import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.preprocessing import preprocess_data

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="EduPro Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

data = preprocess_data()

# =========================
# LOAD MODELS
# =========================

revenue_model = joblib.load(
    "models/revenue_model.pkl"
)

enrollment_model = joblib.load(
    "models/enrollment_model.pkl"
)

# =========================
# TITLE
# =========================

st.title("EduPro Predictive Modeling Dashboard")

# =========================
# KPI CARDS
# =========================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Revenue",
    f"₹ {data['Revenue'].sum():,.0f}"
)

col2.metric(
    "Total Courses",
    len(data)
)

col3.metric(
    "Average Rating",
    round(data['CourseRating'].mean(), 2)
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Prediction Inputs")

category_map = {
    "Programming": 0,
    "Design": 1,
    "Business": 2,
    "Marketing": 3,
    "Data Science": 4,
    "Machine Learning": 5
}

type_map = {
    "Free": 0,
    "Paid": 1
}

level_map = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2
}

selected_category = st.sidebar.selectbox(
    "Course Category",
    list(category_map.keys())
)

selected_type = st.sidebar.selectbox(
    "Course Type",
    list(type_map.keys())
)

selected_level = st.sidebar.selectbox(
    "Course Level",
    list(level_map.keys())
)

category = category_map[selected_category]
course_type = type_map[selected_type]
level = level_map[selected_level]

price = st.sidebar.number_input(
    "Course Price",
    min_value=0.0,
    value=100.0
)

duration = st.sidebar.slider(
    "Course Duration",
    1,
    100,
    20
)

rating = st.sidebar.slider(
    "Course Rating",
    1.0,
    5.0,
    4.0
)

experience = st.sidebar.slider(
    "Teacher Experience",
    0,
    20,
    5
)

teacher_rating = st.sidebar.slider(
    "Teacher Rating",
    1.0,
    5.0,
    4.0
)

# =========================
# INPUT DATA
# =========================

input_data = pd.DataFrame(
    [[
        category,
        course_type,
        level,
        price,
        duration,
        rating,
        experience,
        teacher_rating
    ]],
    columns=[
        'CourseCategory',
        'CourseType',
        'CourseLevel',
        'CoursePrice',
        'CourseDuration',
        'CourseRating',
        'YearsOfExperience',
        'TeacherRating'
    ]
)

# =========================
# PREDICTIONS
# =========================

revenue_prediction = revenue_model.predict(
    input_data
)

enrollment_prediction = enrollment_model.predict(
    input_data
)

st.header("Predictions")

p1, p2 = st.columns(2)

p1.success(
    f"Estimated Revenue: ₹ {revenue_prediction[0]:.2f}"
)

p2.info(
    f"Predicted Enrollments: {int(enrollment_prediction[0])}"
)

# =========================
# EDA
# =========================

st.header("Exploratory Data Analysis")

# Revenue by Category
fig1, ax1 = plt.subplots(figsize=(10,5))

data.groupby('CourseCategory')['Revenue'].sum().plot(
    kind='bar',
    ax=ax1
)

plt.title("Revenue by Category")

st.pyplot(fig1)

# Enrollment Distribution
fig2, ax2 = plt.subplots(figsize=(10,5))

sns.histplot(
    data['EnrollmentCount'],
    bins=20,
    ax=ax2
)

plt.title("Enrollment Distribution")

st.pyplot(fig2)

# Rating vs Revenue
fig3, ax3 = plt.subplots(figsize=(10,5))

sns.scatterplot(
    x=data['CourseRating'],
    y=data['Revenue'],
    ax=ax3
)

plt.title("Course Rating vs Revenue")

st.pyplot(fig3)

# =========================
# FEATURE IMPORTANCE
# =========================

st.header("Feature Importance")

feature_names = [
    'CourseCategory',
    'CourseType',
    'CourseLevel',
    'CoursePrice',
    'CourseDuration',
    'CourseRating',
    'YearsOfExperience',
    'TeacherRating'
]

importance = revenue_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

st.dataframe(importance_df)

fig4, ax4 = plt.subplots(figsize=(10,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df,
    ax=ax4
)

plt.title("Feature Importance")

st.pyplot(fig4)

# =========================
# BUSINESS INSIGHTS
# =========================

st.header("Business Insights")

st.write("""
### Key Findings

1. Course Price strongly influences revenue.

2. Highly rated courses attract more enrollments.

3. Experienced instructors improve learner trust.

4. Premium courses generate higher revenue.

5. Programming and Data Science courses show strong demand.

6. Teacher ratings positively affect enrollments and revenue.
""")