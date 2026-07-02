from django.urls import path
from . import views

urlpatterns = [
    # --- Core AI Stress Prediction Page ---
    path('', views.predict_stress, name='predict_stress'),
]