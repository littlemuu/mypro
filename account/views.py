from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import authenticate,login
from .forms import LoginForm

# Create your views here.
def user_login(request):
    if request.method=='POST':
        form=LoginForm(request.POST)
        if form.is_valid():
            cd=form.cleaned_data
            user=authenticate(request,username=cd['username'],password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request,user)
                    return HttpResponse('Authenticated successfully')
                else:
                    return HttpResponse('无效账户')
            else:
                return HttpResponse('Invalid login')
    else:
        form=LoginForm()
    return render(request,'account/login.html',{'form':form})
                
from django.contrib.auth.decorators import login_required
@login_required
def index(request):
    return render(request,
                  'account/index.html',
                  {'section': 'index'})