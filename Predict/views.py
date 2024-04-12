from django.shortcuts import render
from .models import Prediction

# Create your views here.

def grades_list(request):
    grades=Prediction.objects.order_by('test_time')
    return render(request,'adding.html',{'grades':grades})