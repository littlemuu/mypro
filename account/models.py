from django.db import models
from django.conf import settings

# Create your models here.

class Teacher(models.Model):
    user=models.OneToOneField(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    teacher_id=models.CharField(max_length=50)

    def __str__(self):
        return self.user.username
    
class Classes(models.Model):
    Class_Number_id = models.CharField(max_length=50, primary_key=True)  # 班级号作为主键
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)  # 使用 ForeignKey 关联老师表

    def __str__(self):
        return self.Class_Number_id