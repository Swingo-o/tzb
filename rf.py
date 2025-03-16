import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 读取数据
data = pd.read_csv("xunlian.csv")

# 打乱数据
data_shuffled = data.sample(frac=1, random_state=42)

# 分离特征和目标
X = data_shuffled.drop(columns=['Longitude', 'Latitude', 'depth'])  # 去掉水深
#X = data_shuffled.drop(columns=['depth'])  # 去掉水深
y = data_shuffled['depth']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# 构建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算平均绝对误差
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# 保存模型
import joblib
joblib.dump(model, "random_forest_model.joblib")

# 读取需要预测的新数据
new_data = pd.read_csv("yanzheng.csv")

# 分离特征并进行标准化
#X_new = new_data.drop(columns=['depth'])  # 去掉水深
#X_new = new_data.drop(columns=['Lon', 'Lat', 'depth'])  # 去掉水深
X_new = new_data.drop(columns=['Longitude', 'Latitude', 'depth'])  # 去掉水深
#X_new = new_data.drop(columns=['Longitude', 'Latitude'])  # 去掉水深
#X_new = new_data
X_new_scaled = scaler.transform(X_new)

# 加载模型进行预测
loaded_model = joblib.load("random_forest_model.joblib")
predictions = loaded_model.predict(X_new_scaled)

# 加载现有的CSV文件
existing_data = pd.read_csv('result2.csv')

# 将预测出来的浅水深度数据添加为新列
existing_data['random_forest'] = predictions

# 将更新后的DataFrame保存回CSV文件中
existing_data.to_csv('result2.csv', index=False)
