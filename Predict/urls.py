from django.urls import path
from . import views

app_name='Predict'

urlpatterns = [
    path('',views.grades_list,name='grades_list'),
    path('g_add/',views.add_grades,name='add_grades'),
]
