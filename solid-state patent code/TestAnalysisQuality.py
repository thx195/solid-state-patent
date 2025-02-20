import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from joblib import dump
import os
import json
from collections import Counter

# 创建结果保存目录
results_dir = "./Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']
collection = db['AIPatentData']

# 定义批处理大小
batch_size = 10000

# 初始化全局变量
global_X = []
global_y = []
all_columns = set()

def get_top_categories(column, top_n=10):
    """获取每个类别列中占比最高的前N个类别"""
    counter = Counter(column)
    most_common = counter.most_common(top_n)
    top_categories = [cat for cat, _ in most_common]
    return top_categories

# 随机抽取5万个样本进行编码器的初始化
sample_size = 50000
sample_cursor = collection.aggregate([{"$sample": {"size": sample_size}}])
sample_data = list(sample_cursor)
sample_df = pd.DataFrame(sample_data)

# 确定每个标签变量的前10个占比最高的类别
label_vars = ['申请国家/地区', '公开国家/地区代码', '优先权国家/地区', '公开专利文献类型识别代码', '失效/有效']
top_categories = {var: get_top_categories(sample_df[var], top_n=10) for var in label_vars}

# 仅保留这些前10类别的数据进行OneHotEncoder的fit
for var in label_vars:
    sample_df = sample_df[sample_df[var].isin(top_categories[var])]

encoder = OneHotEncoder(drop='first', handle_unknown='ignore').fit(sample_df[label_vars].fillna('Unknown'))

def preprocess(batch, encoder):

    # 处理标签型变量，将其转换为0-1变量
    label_vars = ['申请国家/地区', '公开国家/地区代码', '优先权国家/地区', '公开专利文献类型识别代码', '失效/有效']
    encoded_labels = encoder.transform(batch[label_vars].fillna('Unknown')).toarray()

    # 修改优先权日的处理方式
    def process_priority_date(date_str):
        try:
            if pd.isna(date_str):
                return np.nan  # 返回NaN
            date_str = str(date_str)
            if ' | ' in date_str:
                date_str = date_str.split(' | ')[0]  # 取分割后的第一部分
            timestamp = pd.to_datetime(date_str, errors='raise').timestamp()
            return timestamp
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            return np.nan
        
    # 将时间字段转换为时间戳的整数表示
    batch['公开日期'] = batch['公开日期'].apply(process_priority_date)
    batch['优先权日'] = batch['优先权日'].apply(process_priority_date)

    # 填充缺失的LDA列
    for i in range(1, 11):
        lda_column = f'LDA{i}'
        if lda_column not in batch.columns:
            batch[lda_column] = 0
        else:
            batch[lda_column] = batch[lda_column].fillna(0)

    for i in ['标题单词数量', '发明人数量', 'IPC 大类数量', 'DWPI 同族专利成员数量', '权利要求 (英语)数量']:
        if i not in batch.columns:
            batch[i] = 0

    # 记录所有可能的列名
    all_columns.update(batch.columns)

    # 合并处理后的数据
    num_vars = ['标题单词数量', '发明人数量', 'IPC 大类数量', 'DWPI 同族专利成员数量', '权利要求 (英语)数量', 
                '引用的参考文献数 - 专利', '引用的参考文献计数 - 非专利', '施引专利计数', 
                'LDA1', 'LDA2', 'LDA3', 'LDA4', 'LDA5', 'LDA6', 'LDA7', 'LDA8', 'LDA9', 'LDA10']
    time_vars = ['公开日期', '优先权日']
    
    X = np.concatenate([batch[num_vars].values, encoded_labels, batch[time_vars].values], axis=1)
    y = batch['DWPI 同族专利成员数量'].values

    # 删除包含NaN的行
    non_nan_indices = ~np.isnan(X).any(axis=1)
    X = X[non_nan_indices]
    y = y[non_nan_indices]

    return X, y

# 批量处理数据并汇总到全局变量中
batch_number = 0

stop = 0

while True:
    cursor = collection.find({}, {'_id': 0}).skip(batch_number * batch_size).limit(batch_size)  # 创建新游标并设置skip和limit
    batch = list(cursor)  # 获取一个批次
    if not batch:
        break

    batch_df = pd.DataFrame(batch)
    if batch_df.empty:
        print("Batch DataFrame is empty. Skipping.")
        continue

    print(f"Processing batch {batch_number} with {len(batch_df)} records.")
    
    X, y = preprocess(batch_df, encoder)
    global_X.append(X)
    global_y.append(y)
    
    batch_number += 1
    
    # stop+=1
    # if stop == 3:
    #     break

# 对所有数据进行统一特征处理，填充缺失的特征列
# 对所有数据进行统一特征处理，填充缺失的特征列
def align_features(X, all_columns):
    """根据所有可能的列，填充X缺少的列"""
    missing_cols = list(all_columns - set(X.columns))
    if missing_cols:
        padding = np.zeros((X.shape[0], len(missing_cols)))
        X = np.hstack((X, padding))
    return X

# 使用新的 align_features 函数，并传递全局列信息
global_X = [align_features(pd.DataFrame(X), all_columns) for X in global_X]

# 将所有批次数据合并
global_X = np.vstack(global_X)
global_y = np.concatenate(global_y)


# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(global_X, global_y, test_size=0.2, random_state=42)

# 在合并所有批次数据后，保存为JSON文件
global_data = {
    "features": global_X.tolist(),
    "labels": global_y.tolist()
}
with open(os.path.join(results_dir, 'processed_data.json'), 'w') as f:
    json.dump(global_data, f, indent=4)

# 在模型训练之前输出样本量信息
print(f"Training sample size: {X_train.shape[0]}")
print(f"Testing sample size: {X_test.shape[0]}")


# 构建并训练模型
models = {
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(n_estimators=100),
    'CausalForest': CausalForestDML(model_y=RandomForestRegressor(), model_t=RandomForestRegressor(), n_estimators=100)
}

results = {}

try:
    for name, model in models.items():
        if name == 'CausalForest':
            T_train = X_train[:, :10]  # 处理变量
            X_train_cf = X_train[:, 10:]  # 控制变量
            
            model.fit(y_train, T_train, X=X_train_cf)
            treatment_effects = model.effect(X_test[:, 10:], T_test=X_test[:, :10])
            results[name] = {
                'Treatment Effects': treatment_effects.tolist()
            }
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            mse_train = mean_squared_error(y_train, y_pred_train)
            mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)

            mse_test = mean_squared_error(y_test, y_pred_test)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            total_y = np.concatenate([y_train, y_test])
            total_pred = np.concatenate([y_pred_train, y_pred_test])
            
            mse_total = mean_squared_error(total_y, total_pred)
            mape_total = mean_absolute_percentage_error(total_y, total_pred)
            r2_total = r2_score(total_y, total_pred)

            results[name] = {
                'train': {
                    'MSE': mse_train,
                    'MAPE': mape_train,
                    'R^2': r2_train,
                },
                'test': {
                    'MSE': mse_test,
                    'MAPE': mape_test,
                    'R^2': r2_test,
                },
                'total': {
                    'MSE': mse_total,
                    'MAPE': mape_total,
                    'R^2': r2_total,
                },
                'coefficients': model.coef_ if hasattr(model, 'coef_') else 'N/A'
            }
        
        # 保存模型
        dump(model, os.path.join(results_dir, f'{name}_model.joblib'))

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 保存全局分析结果，无论是否发生异常
    global_results_path = os.path.join(results_dir, 'global_results.json')
    with open(global_results_path, 'w') as f:
        json.dump(results, f, indent=4)

# 完成处理后关闭数据库连接
client.close()

print("全局数据处理和分析完成，结果已保存到./Results目录下。")
