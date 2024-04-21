import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, k_means
from sklearn.decomposition import PCA
from io import BytesIO
import base64

from sympy import centroid
from .models import Stu_Info  # 导入学生信息模型

def get_student_data(stu):
    student = Stu_Info.objects.get(id=stu)
    student_data = pd.DataFrame({
        'Medu': [student.Medu],
        'Fedu': [student.Fedu],
        'traveltime': [student.traveltime],
        'studytime': [student.studytime],
        'failures': [student.failures],
        'famrel': [student.famrel],
        'freetime': [student.freetime],
        'goout': [student.goout],
        'health': [student.healths],
        'absences': [student.absences],
    })
    return student_data

def get_student_G3(stu):
    student = Stu_Info.objects.get(id=stu)
    student_G3 = pd.DataFrame({
        'G3': [student.G3],
    })
    return student_G3

def generate_kmeans_plot(stu):
    # 读取CSV文件并选择指定的列
    data1 = pd.read_csv(r'D:\college3\111\data\student-mat.csv', sep=';', usecols=['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences'])
    data2 = pd.read_csv(r'D:\college3\111\data\student-por.csv', sep=';', usecols=['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences'])   
    data = pd.concat([data1,data2])
    #处理缺失值
    #data = data3.dropna()
    G3_data1 = pd.read_csv(r'D:\college3\111\data\student-mat.csv', sep=';', usecols=['G3'])
    G3_data2 = pd.read_csv(r'D:\college3\111\data\student-por.csv', sep=';', usecols=['G3'])
    G3_data = pd.concat([G3_data1,G3_data2])

    # 查询当前学生的信息
    student_data = get_student_data(stu)
    # 合并数据集和当前学生的信息
    s_d_p = pd.concat([data, student_data], ignore_index=True)

    # 使用PCA进行线性降维
    pca = PCA(n_components=1)
    s_d_p = pca.fit_transform(s_d_p)
    # 将降维后的数据转换为DataFrame格式
    s_d_p = pd.DataFrame(s_d_p, columns=['PCA'])

    # 查询当前学生的G3信息
    student_G3 = get_student_G3(stu)
    # 合并
    s_G_p = pd.concat([G3_data,student_G3], ignore_index=True)

    data_with_G3 = pd.concat([pd.DataFrame(s_d_p, columns=['PCA']), s_G_p], axis=1)

    # 设置聚类数量
    k = 5

    # 初始化KMeans模型并进行训练
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_with_G3)

    # 获取聚类中心和每个样本所属的簇
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    distance = pd.Series()
    for i in range(len(data_with_G3)):
        Xa = np.array(data_with_G3.iloc[i])
        Xb = centroids[labels[i]]
        distance.at[i] = np.linalg.norm(Xa-Xb)  # 使用at[]方法设置值

    # 设置异常值比例
    Proportion_outlier = 0.01

    # 计算异常值的数量
    number_of_outliers = int(Proportion_outlier*len(distance))

    # 设定异常值的阈值
    threshold = distance.nlargest(number_of_outliers).min()

    # 判断是否为异常值
    data_with_G3['outlier'] = (distance >= threshold).astype(int)

    # 判断当前学生是否为异常
    current_student_outlier = data_with_G3.iloc[-1]['outlier']

    # 可视化聚类结果
    colors = ['r', 'g', 'b', 'c', 'm']  # 设置不同簇的颜色
    for i in range(len(data_with_G3)):
        plt.scatter(data_with_G3.iloc[i, 0], data_with_G3.iloc[i, 1], c=colors[labels[i]], alpha=0.5, marker='o')  # 通过alpha参数调整颜色的透明度

    # 标记聚类中心
    for i in range(k):
        plt.scatter(centroids[i, 0], centroids[i, 1], c='k', marker='*', s=100)

    # 标注当前学生的点
    student_index = len(data_with_G3) - 1
    student_point = (data_with_G3.iloc[student_index, 0], data_with_G3.iloc[student_index, 1])
    plt.scatter(student_point[0], student_point[1], c='red', marker='^', s=200)  # 增加当前学生标记的大小

    plt.xlabel('data')
    plt.ylabel('G3')
    plt.title('KMeans Clustering')

    # 将图像保存为字节流
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # 在编码之前关闭图形对象

    # 将字节流编码为base64字符串
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f'data:image/png;base64,{image_base64}', current_student_outlier
