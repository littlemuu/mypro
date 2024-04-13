from django.shortcuts import redirect, render,get_object_or_404
from .models import Stu_Info
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger
from .km import generate_kmeans_plot

# Create your views here.

def Stu_List(request):
    object_list = Stu_Info.objects.order_by('Class_Number_id', 'Stu_Number')
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
    stu = get_object_or_404(Stu_Info, Stu_Number=SN, Class_Number_id=CN)
    
    # 调用K均值聚类算法函数并获取生成的图片文件路径和异常值检测结果
    image_path, is_outlier = generate_kmeans_plot(stu.id)
    
    # 将图片路径和异常值检测结果添加到上下文数据中
    context = {'stu': stu, 'image_path': image_path, 'is_outlier': is_outlier}
    
    return render(request, 'Students/stu_info/detail.html', context)

def add_student(request):
    if request.method == 'POST':
        # 如果表单被提交，处理表单数据
        stu_number = request.POST.get('Stu_Number')
        class_number = request.POST.get('Class_Number_id')
        medu = request.POST.get('Medu')
        fedu = request.POST.get('Fedu')
        traveltime = request.POST.get('traveltime')
        studytime = request.POST.get('studytime')
        failures = request.POST.get('failures')
        famrel = request.POST.get('famrel')
        freetime = request.POST.get('freetime')
        goout = request.POST.get('goout')
        healths = request.POST.get('healths')
        absences = request.POST.get('absences')
        g1 = request.POST.get('G1')
        g2 = request.POST.get('G2')
        g3 = request.POST.get('G3')
        
        # 创建学生对象并保存到数据库
        Stu_Info.objects.create(
            Stu_Number=stu_number,
            Class_Number_id=class_number,
            Medu=medu,
            Fedu=fedu,
            traveltime=traveltime,
            studytime=studytime,
            failures=failures,
            famrel=famrel,
            freetime=freetime,
            goout=goout,
            healths=healths,
            absences=absences,
            G1=g1,
            G2=g2,
            G3=g3,
            # 设置其他学生信息字段
        )
        # 重定向到学生列表页面
        return redirect('Students:Stu_List')
    else:
        # 如果是GET请求，返回空的表单
        return render(request, 'Students/list.html')
