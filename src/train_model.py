import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

from preprocessing import preprocess_data

# =========================
# Load Data
# =========================

data = preprocess_data()

# =========================
# Features
# =========================

X = data[
    [
        'CourseCategory',
        'CourseType',
        'CourseLevel',
        'CoursePrice',
        'CourseDuration',
        'CourseRating',
        'YearsOfExperience',
        'TeacherRating'
    ]
]

# =====================================================
# REVENUE MODEL
# =====================================================

y_revenue = data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_revenue,
    test_size=0.2,
    random_state=42
)

revenue_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

revenue_model.fit(X_train, y_train)

revenue_predictions = revenue_model.predict(X_test)

print("\nRevenue Model Performance")
print("----------------------------")

print(
    "MAE :",
    mean_absolute_error(y_test, revenue_predictions)
)

print(
    "RMSE :",
    np.sqrt(
        mean_squared_error(
            y_test,
            revenue_predictions
        )
    )
)

print(
    "R2 Score :",
    r2_score(
        y_test,
        revenue_predictions
    )
)

# Save Revenue Model
joblib.dump(
    revenue_model,
    "models/revenue_model.pkl"
)

# =====================================================
# ENROLLMENT MODEL
# =====================================================

y_enrollment = data['EnrollmentCount']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_enrollment,
    test_size=0.2,
    random_state=42
)

enrollment_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

enrollment_model.fit(X_train2, y_train2)

enrollment_predictions = enrollment_model.predict(X_test2)

print("\nEnrollment Model Performance")
print("--------------------------------")

print(
    "MAE :",
    mean_absolute_error(
        y_test2,
        enrollment_predictions
    )
)

print(
    "RMSE :",
    np.sqrt(
        mean_squared_error(
            y_test2,
            enrollment_predictions
        )
    )
)

print(
    "R2 Score :",
    r2_score(
        y_test2,
        enrollment_predictions
    )
)

# Save Enrollment Model
joblib.dump(
    enrollment_model,
    "models/enrollment_model.pkl"
)

print("\nModels Saved Successfully!")