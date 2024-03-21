from django.contrib import admin
from .models import Stu_Info

# Register your models here.

@admin.register(Stu_Info)
class PostAdmin(admin.ModelAdmin):
    list_display=('Class_Number','Stu_Number')
    search_fields=('Class_Number','Stu_Number')