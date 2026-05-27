import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():

    users_path = os.path.join(BASE_DIR, "data", "users.csv")
    teachers_path = os.path.join(BASE_DIR, "data", "teachers.csv")
    courses_path = os.path.join(BASE_DIR, "data", "courses.csv")
    transactions_path = os.path.join(BASE_DIR, "data", "transactions.csv")

    users = pd.read_csv(users_path)
    teachers = pd.read_csv(teachers_path)
    courses = pd.read_csv(courses_path)
    transactions = pd.read_csv(transactions_path)

    return users, teachers, courses, transactions


def preprocess_data():

    users, teachers, courses, transactions = load_data()

    # =========================
    # Merge datasets
    # =========================

    merged_df = pd.merge(
        transactions,
        courses,
        on='CourseID'
    )

    merged_df = pd.merge(
        merged_df,
        teachers,
        on='TeacherID'
    )

    # =========================
    # Enrollment Count
    # =========================

    enrollments = merged_df.groupby(
        'CourseID'
    ).size().reset_index(name='EnrollmentCount')

    # =========================
    # Revenue
    # =========================

    revenue = merged_df.groupby(
        'CourseID'
    )['Amount'].sum().reset_index(name='Revenue')

    # =========================
    # Teacher Features
    # =========================

    teacher_features = merged_df.groupby(
        'CourseID'
    ).agg({
        'YearsOfExperience': 'mean',
        'TeacherRating': 'mean'
    }).reset_index()

    # =========================
    # Final Merge
    # =========================

    course_data = pd.merge(
        courses,
        enrollments,
        on='CourseID',
        how='left'
    )

    course_data = pd.merge(
        course_data,
        revenue,
        on='CourseID',
        how='left'
    )

    course_data = pd.merge(
        course_data,
        teacher_features,
        on='CourseID',
        how='left'
    )

    # =========================
    # Fill Missing Values
    # =========================

    course_data['EnrollmentCount'] = course_data[
        'EnrollmentCount'
    ].fillna(0)

    course_data['Revenue'] = course_data[
        'Revenue'
    ].fillna(0)

    course_data['YearsOfExperience'] = course_data[
        'YearsOfExperience'
    ].fillna(0)

    course_data['TeacherRating'] = course_data[
        'TeacherRating'
    ].fillna(0)

    # =========================
    # Feature Engineering
    # =========================

    def price_band(price):

        if price == 0:
            return "Free"

        elif price < 200:
            return "Low"

        elif price < 400:
            return "Medium"

        else:
            return "High"

    course_data['PriceBand'] = course_data[
        'CoursePrice'
    ].apply(price_band)

    def duration_bucket(duration):

        if duration < 10:
            return "Short"

        elif duration < 30:
            return "Medium"

        else:
            return "Long"

    course_data['DurationBucket'] = course_data[
        'CourseDuration'
    ].apply(duration_bucket)

    def rating_tier(rating):

        if rating >= 4:
            return "Excellent"

        elif rating >= 3:
            return "Good"

        else:
            return "Average"

    course_data['RatingTier'] = course_data[
        'CourseRating'
    ].apply(rating_tier)

    # =========================
    # Label Encoding
    # =========================

    category_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    level_encoder = LabelEncoder()

    course_data['CourseCategory'] = category_encoder.fit_transform(
        course_data['CourseCategory']
    )

    course_data['CourseType'] = type_encoder.fit_transform(
        course_data['CourseType']
    )

    course_data['CourseLevel'] = level_encoder.fit_transform(
        course_data['CourseLevel']
    )

    return course_data