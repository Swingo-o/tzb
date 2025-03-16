import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 读取数据文件
data = pd.read_csv('xunlian.csv')

# 提取特征和目标变量
XI = data['Band_1']
XJ = data['Band_2']
XK = data['Band_3']
XL = data['Band_4']
'''
XM = data['Band_5']
XN = data['Band_6']
XO = data['Band_7']
XP = data['Band_8']
'''
dep = data['depth']

# 处理NaN值
XI.replace([0, np.inf, -np.inf], np.nan, inplace=True)
XI.dropna(inplace=True)
XJ = XJ[XI.index]
XK = XK[XI.index]
XL = XL[XI.index]
'''
XM = XM[XI.index]
XN = XN[XI.index]
XO = XO[XI.index]
XP = XP[XI.index]
'''
dep = dep[XI.index]

# 准备预测数据并进行最小值减法和对数变换

XI= XI- np.min(XI) + 1e-5
XJ = XJ-np.min(XJ) + 1e-5
XK = XK-np.min(XK) + 1e-5
XL =XL- np.min(XL) + 1e-5
'''
XM = XM-np.min(XM) + 1e-5
XN = XN-np.min(XN) + 1e-5
XO = XO-np.min(XO) + 1e-5
XP =XP- np.min(XP) + 1e-5
'''
#dep = dep[XI.index]





# 创建线性回归模型并拟合


#X = np.column_stack((XI, XJ, XK, XL, XM, XN, XO, XP))
X = np.column_stack((XI, XJ, XK, XL))
y = dep.values.reshape(-1, 1)
#XX = np.column_stack((np.ones(XI.shape[0]), X))  # 确保创建的常数项数组形状正确
XX = np.column_stack((np.ones(XI.shape[0]), X))  # 确保创建的常数项数组形状正确
reg = np.linalg.lstsq(XX, y, rcond=None)[0]


# 准备预测数据并进行最小值减法和对数变换
predict_data = pd.read_csv('yanzheng.csv')


def preprocess_and_log_transform(column):
    adjusted = column - np.min(column) + 1e-5  # 避免对数变换中的零值
    return np.log(adjusted)


# 对b1到b8及b11应用处理
#b_columns = [f'Band_{i}' for i in [1, 2, 3, 4, 5, 6, 7, 8]]
b_columns = [f'Band_{i}' for i in [1, 2, 3, 4]]
b_columns_transformed = [preprocess_and_log_transform(predict_data[b]) for b in b_columns]

# 将处理后的数据堆叠为一个特征矩阵，用于预测
X_transformed = np.column_stack(b_columns_transformed)

# 确保只选择用于训练模型的特征，这里我们假设用了7个特征加一个常数项
XX_transformed = np.column_stack((np.ones(X_transformed.shape[0]), X_transformed[:, :8]))

# 使用之前训练好的模型进行深度预测
predicted_depths = XX_transformed.dot(reg)


# 加载现有的CSV文件
existing_data = pd.read_csv('result2.csv')

# 将预测出来的浅水深度数据添加为新列
existing_data['log-linear'] = predicted_depths

# 将更新后的DataFrame保存回CSV文件中
existing_data.to_csv('result2.csv', index=False)

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

# # 加载经纬度和水深数据
# lat = np.loadtxt('lat.txt')
# lon = np.loadtxt('lon.txt')
# ZLm = np.loadtxt('ZLm.txt')
#
# # 创建Miller投影的地图
# plt.figure(figsize=(10, 8))
# m = Basemap(projection='mill', lon_0=0)
#
# # 绘制地图和水深数据
# x, y = m(lon, lat)
# pcm = m.pcolormesh(x, y, ZLm, cmap='jet', shading='flat')
# m.colorbar(pcm, location='right', label='Water Depth (m)')
#
# # 添加地图特征
# m.drawcoastlines()
# m.drawcountries()
# m.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1])
#
# # 添加标题
# plt.title('Lyzenga Model Predicted Water Depth')
#
# # 显示图形
# plt.show()

