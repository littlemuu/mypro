from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate,login
from .forms import LoginForm,UserRegistrationForm,UserEditForm,TeacherEditForm
from .models import Teacher
from django.contrib import messages

# Create your views here.
from django.shortcuts import redirect

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(request, username=cd['username'], password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request, user)
                    # 登录成功后重定向到 index 视图
                    return redirect('index')
                else:
                    return HttpResponse('无效账户')
            else:
                return HttpResponse('Invalid login')
    else:
        form = LoginForm()
    return render(request, 'account/login.html', {'form': form})

                
from django.contrib.auth.decorators import login_required
@login_required
def index(request):
    return render(request,
                  'account/index.html',
                  {'section': 'index'})

def register(request):
    if request.method=='POST':
        user_form=UserRegistrationForm(request.POST)
        if user_form.is_valid():
            new_user=user_form.save(commit=False)
            new_user.set_password(user_form.cleaned_data['password'])
            new_user.save()
            Teacher.objects.create(user=new_user)
            return render(request,'account/register_done.html',{'new_user':new_user})
    else:
        user_form=UserRegistrationForm()
    return render(request,'account/register.html',{'user_form':user_form})

@login_required
def edit(request):
    if request.method=='POST':
        user_form=UserEditForm(instance=request.user,data=request.POST)
        teacher_form=TeacherEditForm(instance=request.user.teacher,data=request.POST,files=request.FILES)
        if user_form.is_valid() and teacher_form.is_valid():
            user_form.save()
            teacher_form.save()
            messages.success(request,'更改成功')
        else:
            messages.error(request,'出错了！')
    else:
        user_form=UserEditForm(instance=request.user)
        teacher_form=TeacherEditForm(instance=request.user.teacher)
    return render(request,'account/edit.html',{'user_form':user_form,'teacher_form':teacher_form})