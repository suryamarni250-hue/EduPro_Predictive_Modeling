import joblib
import pandas as pd

# Load model
model = joblib.load("models/revenue_model.pkl")

def predict_revenue(
    category,
    course_type,
    level,
    price,
    duration,
    rating
):

    input_data = pd.DataFrame(
        [[
            category,
            course_type,
            level,
            price,
            duration,
            rating
        ]],
        columns=[
            'CourseCategory',
            'CourseType',
            'CourseLevel',
            'CoursePrice',
            'CourseDuration',
            'CourseRating'
        ]
    )

    prediction = model.predict(input_data)

    return prediction[0]