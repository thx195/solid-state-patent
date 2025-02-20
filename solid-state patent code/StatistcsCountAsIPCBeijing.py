from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import pymongo

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']  # 数据库名称
collection = db['AIPatentData']  # 集合名称

# 预先筛查：筛选【专利权人/申请人 (原始语言)】字段中包含【北京市】的数据
collection.update_many(
    {
        "专利权人/申请人 (原始语言)": {"$regex": "北京市"}
    },
    {"$set": {"符合条件": True}}
)

# 过滤掉空值和NaN值
collection.update_many(
    {
        "$and": [
            {"申请日期": {"$eq": None}},
            {"符合条件": {"$exists": True, "$eq": True}}
        ],
        "$or": [
            {"申请日期": {"$type": "double", "$eq": float('nan')}}
        ]
    },
    {"$set": {"申请日期": None}}
)

# 添加字段 IPC 优先大类
collection.update_many(
    {"符合条件": {"$exists": True, "$eq": True}},
    [{
        "$set": {
            "IPC优先大类": {
                "$cond": {
                    "if": {
                        "$or": [
                            {"$eq": ["$IPC 大类组", None]},
                            {"$ne": [{"$type": "$IPC 大类组"}, "string"]}
                        ]
                    },
                    "then": None,
                    "else": {"$arrayElemAt": [{"$split": ["$IPC 大类组", ", "]}, 0]}
                }
            }
        }
    }]
)

# 聚合查询1：基于 IPC 优先大类 的年份聚合
pipeline_1 = [
    {
        "$match": {
            "申请日期": {"$ne": None},
            "IPC优先大类": {"$ne": None},
            "符合条件": {"$exists": True, "$eq": True}
        }
    },
    {
        "$addFields": {
            "申请日期_年份": {"$year": "$申请日期"}
        }
    },
    {
        "$group": {
            "_id": {
                "IPC优先大类": "$IPC优先大类",
                "申请日期_年份": "$申请日期_年份"
            },
            "专利数量": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "_id.申请日期_年份": 1
        }
    }
]

# 聚合查询2：基于 IPC 优先大类 和 申请国家/地区 的年份聚合
pipeline_2 = [
    {
        "$match": {
            "申请日期": {"$ne": None},
            "IPC优先大类": {"$ne": None},
            "符合条件": {"$exists": True, "$eq": True}
        }
    },
    {
        "$addFields": {
            "申请日期_年份": {"$year": "$申请日期"}
        }
    },
    {
        "$group": {
            "_id": {
                "IPC优先大类": "$IPC优先大类",
                "申请国家/地区": "$申请国家/地区",
                "申请日期_年份": "$申请日期_年份"
            },
            "专利数量": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "_id.申请日期_年份": 1
        }
    }
]

try:
    result_1 = list(collection.aggregate(pipeline_1))
    result_2 = list(collection.aggregate(pipeline_2))
except pymongo.errors.OperationFailure as e:
    print(f"聚合错误: {e}")
    raise  # 重新抛出错误以便进一步处理

# 将结果1转换为DataFrame
data_1 = []
for entry in result_1:
    data_1.append({
        "IPC优先大类": entry['_id']['IPC优先大类'],
        "申请日期_年份": entry['_id']['申请日期_年份'],
        "专利数量": entry['专利数量']
    })

df_1 = pd.DataFrame(data_1)

# 创建按年份分组的透视表
patent_count_df_1 = df_1.pivot(index='IPC优先大类', columns='申请日期_年份', values='专利数量').fillna(0)

# 将结果2转换为DataFrame
data_2 = []
for entry in result_2:
    data_2.append({
        "IPC优先大类": entry['_id']['IPC优先大类'],
        "申请国家/地区": entry['_id']['申请国家/地区'],
        "申请日期_年份": entry['_id']['申请日期_年份'],
        "专利数量": entry['专利数量']
    })

df_2 = pd.DataFrame(data_2)

# 创建按年份和申请国家/地区分组的透视表
patent_count_df_2 = df_2.pivot_table(index=['IPC优先大类', '申请国家/地区'], columns='申请日期_年份', values='专利数量', fill_value=0)

# 保存为CSV文件
patent_count_df_1.to_csv('./BeijingResults/Beijingpatent_count_by_year_and_IPC.csv')
patent_count_df_2.to_csv('./BeijingResults/Beijingpatent_count_by_year_IPC_and_country.csv')

print("CSV文件生成完毕：'Beijingpatent_count_by_year_and_IPC.csv' 和 'Beijingpatent_count_by_year_IPC_and_country.csv'")
