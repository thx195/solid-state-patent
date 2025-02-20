from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import pymongo

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']  # 数据库名称
collection = db['AIPatentData']  # 集合名称

# 首先过滤掉空值和NaN值
collection.update_many(
    {
        "$or": [
            {"申请日期": {"$eq": None}},
            {"申请日期": {"$type": "double", "$eq": float('nan')}}
        ]
    },
    {"$set": {"申请日期": None}}
)

# 聚合查询
pipeline = [
    {
        "$match": {
            "专利权人/申请人 (原始语言)": {"$regex": "北京市"}
        }
    },
    {
        "$match": {
            "申请日期": {"$ne": None}
        }
    },
    {
        "$addFields": {
            "申请日期_年份": {"$year": "$申请日期"},
            "DWPI同族专利成员_count": {
                "$cond": {
                    "if": {
                        "$or": [
                            {"$eq": ["$DWPI 同族专利成员", None]},
                            {"$ne": [{"$type": "$DWPI 同族专利成员"}, "string"]}
                        ]
                    },
                    "then": 0,
                    "else": {"$size": {"$split": ["$DWPI 同族专利成员", " | "]}}
                }
            }
        }
    },
    {
        "$group": {
            "_id": {
                "申请国家/地区": "$申请国家/地区",
                "申请日期_年份": "$申请日期_年份"
            },
            "专利数量": {"$sum": 1},
            "DWPI同族专利成员_total": {"$sum": "$DWPI同族专利成员_count"}
        }
    },
    {
        "$sort": {
            "_id.申请日期_年份": 1
        }
    }
]

try:
    result = list(collection.aggregate(pipeline))
except pymongo.errors.OperationFailure as e:
    print(f"聚合错误: {e}")
    # 找到导致错误的文档并输出
    error_docs = collection.find({"申请日期": {"$type": "double"}})
    for doc in error_docs:
        print(f"错误文档ID: {doc['_id']}, 申请日期: {doc['申请日期']}")
    raise  # 重新抛出错误以便进一步处理


# 将结果转换为DataFrame
data = []
for entry in result:
    data.append({
        "申请国家/地区": entry['_id']['申请国家/地区'],
        "申请日期_年份": entry['_id']['申请日期_年份'],
        "专利数量": entry['专利数量'],
        "DWPI同族专利成员_total": entry['DWPI同族专利成员_total']
    })

df = pd.DataFrame(data)

# 创建按年份分组的透视表
patent_count_df = df.pivot(index='申请国家/地区', columns='申请日期_年份', values='专利数量').fillna(0)
dwpi_total_df = df.pivot(index='申请国家/地区', columns='申请日期_年份', values='DWPI同族专利成员_total').fillna(0)

# 保存为CSV文件
patent_count_df.to_csv('./BeijingResults/Beijing_patent_count_by_year_and_country.csv')
dwpi_total_df.to_csv('./BeijingResults/Beijing_dwpi_total_by_year_and_country.csv')

print("CSV文件生成完毕：'Beijing_patent_count_by_year_and_country.csv' 和 'Beijing_dwpi_total_by_year_and_country.csv'")
