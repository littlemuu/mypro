from django.urls import path
from . import views
import account.urls

app_name = 'Students'

urlpatterns = [
    path('', views.Stu_List, name='Stu_List'),
    path('<str:SN>/<str:CN>/', views.Stu_Detail, name='Stu_Detail'),
    path('add/', views.add_student, name='add_student'), 
    path('delete/<str:SN>/<str:CN>/',views.delete_student,name='delete_student'),
]
