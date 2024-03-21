from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

# Create your models here.

class Stu_Info(models.Model):
    Stu_Number=models.CharField(max_length=50)
    Class_Number=models.CharField(max_length=50)

    def get_absolute_url(self):
        return reverse('Students:Stu_Detail',args=[self.Stu_Number,self.Class_Number])
    