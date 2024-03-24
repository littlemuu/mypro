from django.shortcuts import render,get_object_or_404
from .models import Stu_Info
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger
from .km import generate_kmeans_plot

# Create your views here.

def Stu_List(request):
    object_list = Stu_Info.objects.order_by('Class_Number', 'Stu_Number')
    paginator = Paginator(object_list, 10)
    page = request.GET.get('page')
    try:
        stus = paginator.page(page)
    except PageNotAnInteger:
        stus = paginator.page(1)
    except EmptyPage:
        stus = paginator.page(paginator.num_pages)
    return render(request, 'Students/stu_info/list.html', {'page': page, 'stus': stus})


def Stu_Detail(request, SN, CN):
    stu = get_object_or_404(Stu_Info, Stu_Number=SN, Class_Number=CN)
    
    # 调用K均值聚类算法函数并获取生成的图片文件路径
    image_path = generate_kmeans_plot(stu.id)
    
    # 将图片路径添加到上下文数据中
    context = {'stu': stu, 'image_path': image_path}
    
    return render(request, 'Students/stu_info/detail.html', context)