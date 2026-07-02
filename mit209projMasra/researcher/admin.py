from django.contrib import admin
from .models import PredictionHistory

# --- Register the Student Stress Prediction History Model ---
@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    # This determines which columns show up in your Django Admin dashboard table
    list_display = (
        'id', 
        'student_type', 
        'sleep_hours', 
        'study_hours', 
        'predicted_stress', 
        'timestamp'
    )
    
    # This adds a sidebar filter on the right side
    list_filter = ('predicted_stress', 'student_type', 'month')
    
    # This adds a search box at the top
    search_fields = ('id', 'predicted_stress')