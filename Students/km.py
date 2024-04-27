import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, k_means
#from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
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
        'G3': [student.G3],
    })
    return student_data

def generate_kmeans_plot(stu):
    # 读取CSV文件并选择指定的列
    data1 = pd.read_csv(r'D:\college3\111\data\student-mat.csv', sep=';', usecols=['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences','G3'])
    data2 = pd.read_csv(r'D:\college3\111\data\student-por.csv', sep=';', usecols=['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences','G3'])   
    data = pd.concat([data1,data2])
    #处理缺失值
    
    # 查询当前学生的信息
    student_data = get_student_data(stu)
    # 合并数据集和当前学生的信息
    s_d_p = pd.concat([data, student_data], ignore_index=True)

    # 使用t-SNE进行非线性降维
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(s_d_p)

    # 将降维后的数据转换为DataFrame格式
    data = pd.DataFrame(data_tsne, columns=['t-SNE1', 't-SNE2'])

    # 设置聚类数量
    k = 10

    # 初始化KMeans模型并进行训练
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)

    # 获取聚类中心和每个样本所属的簇
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 计算各点与最近聚类中心的距离
    distance = pd.Series()
    for i in range(len(data)):
        Xa = np.array(data.iloc[i])
        Xb = centroids[labels[i]]
        distance.at[i] = np.linalg.norm(Xa - Xb)

    # 设置异常值比例
    Proportion_outlier = 0.01

    # 计算异常值的数量
    number_of_outliers = int(Proportion_outlier * len(distance))

    # 设定异常值的阈值
    threshold = distance.nlargest(number_of_outliers).min()

    # 判断是否为异常值
    outliers1 = data[distance >= threshold]

    # 计算每个聚类点到其他聚类点的平均距离
    avg_distances = []
    for i in range(k):
        dists = cdist(centroids[i].reshape(1, -1), centroids[np.arange(k) != i], metric='euclidean')
        avg_distance = np.mean(dists)
        avg_distances.append(avg_distance)

    # 找出离其他聚类很远的聚类索引
    outlier_cluster_index = np.argmax(avg_distances)

    # 标记该聚类中的点为异常值
    outliers2 = data[labels == outlier_cluster_index]

    # 合并异常值列表
    outliers = pd.concat([outliers1, outliers2])

    # 判断当前学生是否为异常
    current_student_index = len(data) - 1
    current_student_outlier = current_student_index in outliers.index

    # 可视化聚类结果
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'lime']  # 设置不同簇的颜色
    for i in range(len(data)):
        # 标记异常点为"x"
        if i in outliers.index:
            plt.scatter(data.iloc[i, 0], data.iloc[i, 1], c='k', alpha=0.5, marker='x')
        else:
            plt.scatter(data.iloc[i, 0], data.iloc[i, 1], c=colors[labels[i]], alpha=0.5, marker='o')

    # 标记聚类中心
    for i in range(k):
        plt.scatter(centroids[i, 0], centroids[i, 1], c='k', marker='*', s=100)

    # 标注当前学生的点
    plt.scatter(data.iloc[current_student_index, 0], data.iloc[current_student_index, 1], c='red', marker='^', s=200)  # 增加当前学生标记的大小

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.title('KMeans聚类-t-SNE')

    # 将图像保存为字节流
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # 在编码之前关闭图形对象

    # 将字节流编码为base64字符串
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f'data:image/png;base64,{image_base64}', current_student_outlier
