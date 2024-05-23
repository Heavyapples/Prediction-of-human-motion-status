import tkinter as tk
from tkinter import ttk
import pickle

# 特征名称列表
feature_names = [
    "心率",
    "传感器1-温度",
    "传感器1-一型加速度X",
    "传感器1-一型加速度Y",
    "传感器2-一型加速度Y",
    "传感器2-一型加速度X",
]

# 示例预测函数
def predict_motion_state(feature_values):
    # 加载模型
    with open("Decision Tree_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 使用模型进行预测
    result = model.predict([feature_values])

    return f"运动状态: {result[0]}"

# 点击检测按钮时执行的函数
def on_detect_click():
    feature_values = []
    for entry in feature_entries:
        value = float(entry.get())
        feature_values.append(value)
    result = predict_motion_state(feature_values)
    result_var.set(f"分类结果: {result}")

# 创建主窗口
window = tk.Tk()
window.title("人体运动状态预测系统")

# 添加特征输入框和标签
feature_entries = []
n_features = len(feature_names)
n_columns = 2
n_rows = n_features // n_columns
for i, feature_name in enumerate(feature_names):
    row = i % n_rows + 1
    column = i // n_rows * 2
    label = ttk.Label(window, text=feature_name)
    label.grid(row=row, column=column, padx=5, pady=5)
    entry = ttk.Entry(window)
    entry.grid(row=row, column=column + 1, padx=5, pady=5)
    feature_entries.append(entry)

# 添加检测按钮
detect_button = ttk.Button(window, text="检测", command=on_detect_click)
detect_button.grid(row=n_rows + 1, column=0, padx=5, pady=5)

# 添加分类结果标签
result_var = tk.StringVar()
result_label = ttk.Label(window, textvariable=result_var)
result_label.grid(row=n_rows + 1, column=1, padx=5, pady=5)

# 运行主循环
window.mainloop()
