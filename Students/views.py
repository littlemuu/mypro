from django.shortcuts import render,get_object_or_404
from .models import Stu_Info
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger

# Create your views here.

def Stu_List(request):
    object_list=Stu_Info.objects.all()
    paginator=Paginator(object_list,10)
    page=request.GET.get('page')
    try:
        stus=paginator.page(page)
    except PageNotAnInteger:
        stus=paginator.page(1)
    except EmptyPage:
        stus=paginator.page(paginator.num_pages)
    return render(request,'Students/stu_info/list.html',{'page':page,'stus':stus})

def Stu_Detail(request,SN,CN):
    stu=get_object_or_404(Stu_Info,Stu_Number=SN,Class_Number=CN)
    return render(request,'Students/stu_info/detail.html',{'stu':stu})