from django import forms
from django.contrib.auth.models import User
from .models import Teacher,Classes

class LoginForm(forms.Form):
    username=forms.CharField()
    password=forms.CharField(widget=forms.PasswordInput)

class UserRegistrationForm(forms.ModelForm):
    password=forms.CharField(label='Password',widget=forms.PasswordInput)
    password2=forms.CharField(label='Repeat Password',widget=forms.PasswordInput)

    class Meta:
        model=User
        fields=('username','email')

    def clean_password2(self):
        cd=self.cleaned_data
        if cd['password']!=cd['password2']:
            raise forms.ValidationError('Password don\'t match.')
        return cd['password2']
    
class UserEditForm(forms.ModelForm):
    class Meta:
        model=User
        fields=('username','email')

class TeacherEditForm(forms.ModelForm):
    class Meta:
        model=Teacher
        fields=('teacher_id',)