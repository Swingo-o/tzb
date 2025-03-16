import numpy as np
import pandas as pd
from scipy.stats import linregress

# 从 image.csv 文件中读取数据
image_data = pd.read_csv('xunlian.csv')
# 提取 r2 和 r3 列的数据
r2_data = image_data['Band_1'].values
r3_data = image_data['Band_2'].values

# 提取 depth 列的数据作为目标值
dep_data = image_data['depth'].values
# Stumpf method for SDB (Blue/Green)
a = np.log(1000 * np.pi * r2_data)  # blue
b = np.log(1000 * np.pi * r3_data)  # green
c = a / b
Cbg = c

# 清除Cbg中的0值
Cbg[Cbg == 0] = np.nan

# 构建特征矩阵
C11 = np.column_stack((np.ones_like(Cbg), Cbg))

# 将目标向量重塑为列向量
dep_data_column = dep_data.reshape(-1, 1)

# 线性回归拟合
reg = linregress(C11[:, 1], dep_data_column.flatten())

# 提取回归系数
a0 = reg.intercept
a1 = reg.slope

# 计算预测值
ZSbg = a0 + a1 * Cbg

# 计算相关性
R = np.corrcoef(dep_data, Cbg, rowvar=False)[0, 1]
RR2 = R**2

print('R2 (blue-green) =', RR2)
print('SDB_Stumpf(bg) =', round(a1, 2), ' x pSDB_bg +', round(a0, 2))

# 读取新数据
new_data = pd.read_csv('yanzheng.csv')

# 提取 r2 和 r3 列的数据
r2_data = new_data['Band_1'].values
r3_data = new_data['Band_2'].values

# Stumpf method for SDB (Blue/Green)
a = np.log(1000 * np.pi * r2_data)  # blue
b = np.log(1000 * np.pi * r3_data)  # green
c = a / b
Cbg = c

# 构建特征矩阵
C11 = np.column_stack((np.ones_like(Cbg), Cbg))

# 使用之前训练好的模型参数进行预测
predicted_depth = a0 + a1 * Cbg

# 加载现有的CSV文件
XXXX = pd.read_csv('yanzheng.csv')#3891*5447
existing_data = XXXX.drop(columns=['Band_1','Band_2','Band_3','Band_4'])  # 去掉水深
# 将预测出来的浅水深度数据添加为新列
existing_data['stumpf'] = predicted_depth

# 将更新后的DataFrame保存回CSV文件中
existing_data.to_csv('result2.csv', index=False)
