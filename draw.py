import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取数据
features = pd.read_csv("features.csv", header=None)
labels = pd.read_csv("labels.csv", header=None)

# 特征名称列表
feature_names = [
    "时间戳",
    "心率",
    "传感器1-温度",
    "传感器1-一型加速度X",
    "传感器1-一型加速度Y",
    "传感器1-一型加速度Z",
    "传感器1-二型加速度X",
    "传感器1-二型加速度Y",
    "传感器1-二型加速度Z",
    "传感器1-陀螺仪X",
    "传感器1-陀螺仪Y",
    "传感器1-陀螺仪Z",
    "传感器1-磁场X",
    "传感器1-磁场Y",
    "传感器1-磁场Z",
]

# 添加传感器2和传感器3的标签
for sensor_num in range(2, 4):
    for feature in ["温度", "一型加速度X", "一型加速度Y", "一型加速度Z", "二型加速度X", "二型加速度Y", "二型加速度Z", "陀螺仪X", "陀螺仪Y", "陀螺仪Z", "磁场X", "磁场Y", "磁场Z"]:
        feature_names.append(f"传感器{sensor_num}-{feature}")

features.columns = feature_names

# 计算相关性
correlations = features.corrwith(labels.squeeze(), axis=0)

# 计算特征之间的相关性矩阵
corr_matrix = features.corr()

# 绘制相关性矩阵热力图
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=feature_names, yticklabels=feature_names, annot_kws={'size': 8})
plt.title("特征相关性矩阵热力图", fontproperties=font)
plt.show()

# 使用Seaborn库中的pairplot函数绘制相关性图
strong_correlations = correlations[abs(correlations) > 0.2]  # 您可以根据需要修改阈值
strong_correlation_indices = strong_correlations.index
sns.pairplot(data=features.loc[:, strong_correlation_indices], diag_kind=None)
plt.show()

# 使用matplotlib.pyplot模块输出运动状态分析占比饼状图
label_counts = labels.value_counts()
label_names = [feature_names[index[0]] for index in label_counts.index]

plt.pie(label_counts, labels=label_names, autopct="%1.1f%%")
plt.title("运动状态分析占比", fontproperties=font)
plt.show()

# 计算所有特征的平均值（去除时间戳）
feature_means = features.loc[:, feature_names[1:]].mean()

# 绘制雷达图（去除时间戳）
radar_labels = feature_names[1:]

num_vars = len(feature_means)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

data = feature_means.tolist()
data += data[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, data, color='blue', linewidth=2, label='特征均值')
ax.fill(angles, data, color='blue', alpha=0.25)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])  # 不包括最后一个重复的角度
ax.set_thetagrids(np.degrees(angles[:-1]), labels=radar_labels, fontproperties=font)
plt.title("雷达图", fontproperties=font)
plt.legend(loc="upper right", prop=font)
plt.show()
