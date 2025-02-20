import os
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']  # 数据库名称
collection = db['AIPatentData']  # 集合名称

# 设置数据文件夹路径
data_folder = './OriginalData'

# 获取所有Excel文件的路径
excel_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.xlsx', '.xls'))]

# 遍历并处理每个Excel文件
for file_path in excel_files:
    # 读取Excel文件，跳过第一行
    df = pd.read_excel(file_path, skiprows=1)

    # 将DataFrame转换为字典列表
    data_records = df.to_dict('records')

    try:
        # 批量插入到MongoDB，并设置ordered为False以继续插入尽管出现错误
        collection.insert_many(data_records, ordered=False)
    except BulkWriteError as bwe:
        print(f"Bulk write error occurred for file {file_path}: ", bwe.details)

    # 删除DataFrame以释放内存
    del df

    print(f"数据从 {file_path} 已成功导入MongoDB。")

# 创建索引以避免插入重复数据，这里有多个字段可以组合唯一标识一个记录
collection.create_index([("公开号", 1), ("申请号", 1)], unique=True)

print("数据已成功导入MongoDB，并设置了索引以避免重复。")
