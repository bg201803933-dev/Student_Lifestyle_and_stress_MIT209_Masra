from django.db import models

class PredictionHistory(models.Model):
    student_type = models.IntegerField()
    sleep_hours = models.FloatField()
    study_hours = models.FloatField()
    social_media_hours = models.FloatField()
    attendance = models.FloatField()
    exam_pressure = models.FloatField()
    family_support = models.FloatField()
    month = models.IntegerField()
    predicted_stress = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True) # Automatically saves the time of prediction