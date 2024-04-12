from django.urls import path
from . import views

app_name = 'Students'

urlpatterns = [
    path('', views.Stu_List, name='Stu_List'),
    path('<int:SN>/<int:CN>/', views.Stu_Detail, name='Stu_Detail'),
    path('add/', views.add_student, name='add_student'), 
]
