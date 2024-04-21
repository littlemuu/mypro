from django.shortcuts import render,redirect
from .models import Prediction
from django.contrib.auth.decorators import login_required
from .rnn import draw1,draw2

# Create your views here.

@login_required
def grades_list(request):
    grades=Prediction.objects.order_by('test_time')
    return render(request,'Pre/adding.html',{'grades':grades,'section':'grades_list'})

@login_required
def add_grades(request):
    if request.method == 'POST':
        # 如果表单被提交，处理表单数据
        test_time = request.POST.get('test_time')
        avg_grade = request.POST.get('avg_grade')
        
        # 创建grade对象并保存到数据库
        Prediction.objects.create(
            test_time=test_time,
            avg_grade=avg_grade,
        )
        # 重定向到grades列表页面
        return redirect('Predict:grades_list')
    else:
        # 如果是GET请求，返回空的表单
        return render(request, 'Predict/adding.html')

@login_required
def pre_result(request):
    image_path1 = draw1()
    image_path2 = draw2()
    
    # 将图片路径添加到上下文数据中
    context = {'image_path1': image_path1, 'image_path2': image_path2}
    
    return render(request, 'Pre/result.html', context)