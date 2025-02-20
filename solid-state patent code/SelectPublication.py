from pymongo import MongoClient

# MongoDB连接设置
client = MongoClient('localhost', 27017)
db = client['PatentData']  # 数据库名称
collection = db['AIPatentData']  # 集合名称

# 设置要查询的特定公开号
specific_publication_id = "WO2020253180A1"  # 将此处替换为实际的公开号

# 查询公开号为特定ID的数据
result = collection.find_one({"公开号": specific_publication_id})

# 检查是否找到了匹配的文档
if result:
    print("查询结果：")
    print(result)
else:
    print(f"未找到公开号为 {specific_publication_id} 的记录。")
