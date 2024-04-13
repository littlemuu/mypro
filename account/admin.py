from django.contrib import admin
from .models import Teacher,Classes

# Register your models here.
@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    list_display=['user','teacher_id']

@admin.register(Classes)
class ClassesAdmin(admin.ModelAdmin):
    list_display=['Class_Number_id','teacher']