from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.

class Prediction(models.Model):
    test_time=models.DateTimeField(primary_key=True)
    avg_grade=models.FloatField()