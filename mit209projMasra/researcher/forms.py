from django import forms

class StressPredictionForm(forms.Form):
    # --- Dropdown Menus (Categorical Data) ---
    STUDENT_TYPE_CHOICES = [
        (0, 'School'),
        (1, 'College')
    ]
    student_type = forms.ChoiceField(
        choices=STUDENT_TYPE_CHOICES, 
        label="Are you in School or College?"
    )

    # --- Number Inputs (Floats for hours and percentages) ---
    sleep_hours = forms.FloatField(
        min_value=0.0, 
        max_value=24.0, 
        label="Average Sleep Hours per Day",
        help_text="e.g., 7.5"
    )
    study_hours = forms.FloatField(
        min_value=0.0, 
        max_value=24.0, 
        label="Average Study Hours per Day",
        help_text="e.g., 4.0"
    )
    social_media_hours = forms.FloatField(
        min_value=0.0, 
        max_value=24.0, 
        label="Average Social Media Hours per Day",
        help_text="e.g., 2.5"
    )
    attendance = forms.FloatField(
        min_value=0.0, 
        max_value=100.0, 
        label="Class Attendance Percentage (%)",
        help_text="e.g., 85.5"
    )

    # --- Sliders / Scales (Assuming 1-10 scales based on the dataset metrics) ---
    exam_pressure = forms.FloatField(
        min_value=1.0, 
        max_value=10.0, 
        label="Exam Pressure Level (1 - 10)"
    )
    family_support = forms.FloatField(
        min_value=1.0, 
        max_value=10.0, 
        label="Family Support Level (1 - 10)"
    )
    
    # --- Integer Input ---
    month = forms.IntegerField(
        min_value=1, 
        max_value=12, 
        label="Current Month (1-12)",
        help_text="1 = Jan, 12 = Dec"
    )