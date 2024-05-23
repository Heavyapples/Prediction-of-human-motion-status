import os
import numpy as np
import pandas as pd

def load_data(user_folder, feature_file, label_file, num_samples=2000):
    dtype_dict = {i: np.float32 for i in range(1, 42)}
    na_values = ['?']
    features = pd.read_csv(os.path.join(user_folder, feature_file), header=None, dtype=dtype_dict, na_values=na_values)
    labels = pd.read_csv(os.path.join(user_folder, label_file), header=None)

    # 删除包含缺失值的行
    features.dropna(inplace=True)
    labels.dropna(inplace=True)

    # 随机挑选2000组数据
    random_indices = np.random.choice(features.index, size=num_samples, replace=False)
    features_sampled = features.loc[random_indices]
    labels_sampled = labels.loc[random_indices]

    return features_sampled, labels_sampled


dataset_folder = "C:\\Users\\13729\\Desktop\\运动预测\\运动预测\\dataset"
user_folders = ['A', 'B', 'C', 'D', 'E']
num_samples = 2000

all_features = []
all_labels = []

for user in user_folders:
    user_folder = os.path.join(dataset_folder, user)
    feature_file = f"{user.lower()}.feature"
    label_file = f"{user.lower()}.label"
    features, labels = load_data(user_folder, feature_file, label_file, num_samples)
    all_features.append(features)
    all_labels.append(labels)

# 合并所有用户的数据
all_features = pd.concat(all_features, axis=0)
all_labels = pd.concat(all_labels, axis=0)

# 处理缺失值
all_features = all_features.fillna(all_features.mean())
all_labels = all_labels.fillna(all_labels.mean())

# 重置索引
all_features.reset_index(drop=True, inplace=True)
all_labels.reset_index(drop=True, inplace=True)

# 存储特征数据
all_features.to_csv("features.csv", index=False, header=False)

# 存储标签数据
all_labels.to_csv("labels.csv", index=False, header=False)
