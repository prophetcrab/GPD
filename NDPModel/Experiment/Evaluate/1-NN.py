import numpy as np
import os
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from DataUtils.DataProcessUtils import *
from DataUtils.EvaluateUtils import *



# 读取数据函数
def load_data(folder_path, label, repeat_times=1):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            points = np.loadtxt(file_path, delimiter=',')
            if label == 1:
                #points = voxel_downsampling(points, voxel_size=0.001)
                points = random_sampling(points, target_num_points=2048)
            # 重复数据 repeat_times 次
            for _ in range(repeat_times):
                data.append(points[:, :3])  # 只取前三列（真实数据只保留前三位）
                labels.append(label)
    return np.array(data), np.array(labels)

# # 加载生成数据和真实数据
gen_data_folder = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Samples\Gen_Data"
real_data_folder = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Samples\Real_Data"

X_gen, y_gen = load_data(gen_data_folder, label=0)  # 生成数据label为0
X_real, y_real = load_data(real_data_folder, label=1, repeat_times=1)  # 真实数据label为1，重复7次以匹配样本量

# 合并数据
X = np.vstack([X_gen, X_real])
y = np.hstack([y_gen, y_real])


# 将数据集按 80% 训练集和 20% 测试集进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=46, random_state=42)




print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# 1-NN分类器
class OneNNClassifierWithCD:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point_cloud in X_test:
            distances = [chamfer_distance(test_point_cloud, train_point_cloud) for train_point_cloud in self.X_train]
            nearest_neighbor_index = np.argmin(distances)
            predictions.append(self.y_train[nearest_neighbor_index])
        return np.array(predictions)

# 创建1-NN分类器
classifier = OneNNClassifierWithCD()
classifier.fit(X_train, y_train)

my_X, my_y = load_data(gen_data_folder, label=0)
# 预测测试集的标签
y_pred = classifier.predict(my_X)

# 计算分类准确率
accuracy = accuracy_score(my_y, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# 输出分类报告（包括精度、召回率和 F1 分数）
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

