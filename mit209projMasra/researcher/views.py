import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from .models import PredictionHistory # Ensure you have this in your models.py

# -------------------------------------------------------------------
# LOAD THE MODEL ONCE WHEN THE SERVER STARTS
# -------------------------------------------------------------------
# Update this path based on exactly where you put the .pkl file in your Django project.
# Using settings.BASE_DIR ensures Django finds it no matter where you run the server.
MODEL_PATH = os.path.join(settings.BASE_DIR, 'best_stress_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("Machine Learning model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading the machine learning model: {e}")

# -------------------------------------------------------------------
# PREDICTION VIEW
# -------------------------------------------------------------------
def predict_stress(request):
    # Fetch the history of predictions (last 10 entries) from the database
    history = PredictionHistory.objects.all().order_by('-timestamp')[:10]
    
    if request.method == 'POST':
        # Safety check in case the model file is missing
        if model is None:
            return render(request, 'predict.html', {
                'error': 'The machine learning model could not be found or loaded.',
                'history': history
            })
            
        try:
            # 1. Extract data submitted from the HTML form
            # Convert strings from POST request to appropriate numeric types
            student_type = int(request.POST.get('student_type')) # 0 for School, 1 for College
            sleep_hours = float(request.POST.get('sleep_hours'))
            study_hours = float(request.POST.get('study_hours'))
            social_media_hours = float(request.POST.get('social_media_hours'))
            attendance = float(request.POST.get('attendance'))
            exam_pressure = float(request.POST.get('exam_pressure'))
            family_support = float(request.POST.get('family_support'))
            month = int(request.POST.get('month', 1)) # Default to month 1 if empty

            # 2. Format the data precisely as the model expects it (Pandas DataFrame)
            # The column names MUST match the columns in your original training dataset exactly.
            input_data = pd.DataFrame({
                'Student_Type': [student_type],
                'Sleep_Hours': [sleep_hours],
                'Study_Hours': [study_hours],
                'Social_Media_Hours': [social_media_hours],
                'Attendance': [attendance],
                'Exam_Pressure': [exam_pressure],
                'Family_Support': [family_support],
                'Month': [month]
            })

            # 3. Make the prediction
            prediction = model.predict(input_data)[0]

            # 4. Save the inputs and the prediction to the database for History
            PredictionHistory.objects.create(
                student_type=student_type,
                sleep_hours=sleep_hours,
                study_hours=study_hours,
                social_media_hours=social_media_hours,
                attendance=attendance,
                exam_pressure=exam_pressure,
                family_support=family_support,
                month=month,
                predicted_stress=prediction
            )

            # Re-fetch history so the new prediction shows up immediately
            updated_history = PredictionHistory.objects.all().order_by('-timestamp')[:10]

            # 5. Return the result back to the template
            return render(request, 'predict.html', {
                'prediction': prediction,
                'history': updated_history
            })
            
        except Exception as e:
            # If the user submitted bad data (e.g., letters instead of numbers)
            return render(request, 'predict.html', {
                'error': f'An error occurred during prediction: {str(e)}',
                'history': history
            })

    # If request is GET (user simply visits the page)
    return render(request, 'predict.html', {'history': history})