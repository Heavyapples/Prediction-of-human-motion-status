import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 读取数据
features = pd.read_csv("features.csv", header=None)
labels = pd.read_csv("labels.csv", header=None)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 定义分类器
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=2000)
}

metrics_data = []

# 训练并评估每个分类器
for classifier_name, classifier in classifiers.items():
    print(f"=== {classifier_name} ===")

    # 训练分类器
    classifier.fit(X_train, y_train.values.ravel())

    # 对测试集进行预测
    y_pred = classifier.predict(X_test)

    # 输出分类报告
    print(classification_report(y_test, y_pred))

    # 计算并存储每个分类器的精确率、召回率、F1分数
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    metrics_data.append([precision, recall, f1_score])

# 绘制柱状图对比各个分类器的性能
metrics = ['Precision', 'Recall', 'F1-score']
n_metrics = len(metrics)
n_classifiers = len(classifiers)
classifier_names = list(classifiers.keys())

fig, ax = plt.subplots(figsize=(10, 5))

index = np.arange(n_metrics)
bar_width = 0.15

for i, (classifier_name, data) in enumerate(zip(classifier_names, metrics_data)):
    ax.bar(index + i * bar_width, data, bar_width, label=classifier_name)

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Classifier Metrics Comparison')
ax.set_xticks(index + bar_width * (n_classifiers - 1) / 2)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()
