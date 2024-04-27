from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.shortcuts import redirect, render
from account.models import Classes

# Create your models here.
class Stu_Info(models.Model):
    Stu_Number = models.CharField(max_length=50)
    Class_Number = models.ForeignKey(Classes, on_delete=models.CASCADE)
    Medu = models.IntegerField()  # Mother's education level
    Fedu = models.IntegerField()  # Father's education level
    traveltime = models.IntegerField()  # Home to school travel time
    studytime = models.IntegerField()  # Weekly study time
    failures = models.IntegerField()  # Number of past class failures
    famrel = models.IntegerField()  # Quality of family relationships
    freetime = models.IntegerField()  # Free time after school
    goout = models.IntegerField()  # Going out with friends
    healths = models.IntegerField()  # Current health status
    absences = models.IntegerField()  # Number of school absences
    G1 = models.IntegerField(null=True, default=None)
  # First period grade
    G2 = models.IntegerField(null=True, default=None)  # Second period grade
    G3 = models.IntegerField()  # Final grade

    def get_absolute_url(self):
        return reverse('Students:Stu_Detail', args=[self.Stu_Number, self.Class_Number])

    
    